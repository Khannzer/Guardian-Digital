# Herramientas principales de Flask
from flask import Flask, render_template, jsonify, request, redirect, url_for, session
# Importando la funcion download_embeddings desde helper.py
from src.helper import download_embeddings
# Prompt del sistema para guiar a la IA en su respuesta
from src.prompt import system_prompt
# Framework de orquestacion - LangChain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
# Estructura de datos para la respuesta de la IA
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv
# Conexion BD
from conexionDb.conexionDb import ConexionDb
# Seguridad de contraseñas
from werkzeug.security import check_password_hash, generate_password_hash
# Ejecuta 2 tareas en paralelo (detección de emoción y transcripción de audio) 
from concurrent.futures import ThreadPoolExecutor
import tempfile
import logging
import os
import warnings
from openai import OpenAI # transcripcion de voz
import librosa  # lee y procesa archivos de audio para HuBERT
from transformers import pipeline # carga el modelo HuBERT para detección de emociones en voz
from twilio.rest import Client as TwilioClient # cliente para enviar SMS con Twilio
from typing import Optional, Literal
import smtplib
import secrets
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta


warnings.filterwarnings("ignore")

# Configuración del sistema para que cada mensaje muestre: fecha, nivel y el mensaje
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Intenta cargar el modelo Hubert al arrancar la app, si falla la app no se caera
detector_emociones = None
try:
    logger.info("Cargando modelo HuBERT de emociones...")
    detector_emociones = pipeline("audio-classification", model="superb/hubert-large-superb-er")
    logger.info("✅ Modelo HuBERT listo.")
except Exception as e:
    logger.warning(f"⚠️ Modelo HuBERT no disponible: {e}. La app seguirá funcionando sin detección de emoción por voz.")

# Creamos la app de Flask y cargamos las variables de entorno desde el .env
app = Flask(__name__)
load_dotenv()

TWILIO_SID    = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN  = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_NUMERO = os.getenv("TWILIO_PHONE_NUMBER")

app.secret_key        = os.getenv("FLASK_SECRET_KEY") # encriptacion de sesiones de lo usuarios
PINECONE_API_KEY      = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY        = os.getenv("OPENAI_API_KEY")
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB máximo por archivo

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"]   = OPENAI_API_KEY

client_openai = OpenAI(api_key=OPENAI_API_KEY) # se usa unicamente para Whisper

GMAIL_USER     = os.getenv("GMAIL_USER")
GMAIL_PASSWORD = os.getenv("GMAIL_PASSWORD")
APP_URL        = os.getenv("APP_URL", "http://localhost:8080")

# convierte texto en vectores para buscar el pinecode
embeddings = download_embeddings()
# conexion al indice en pinecode
docsearch  = PineconeVectorStore.from_existing_index(
    index_name="guardian-digital",
    embedding=embeddings
)
# cuando llega un mensaje, busca 2 fragmentos mas similiares
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# Esquema de la respuesta que la IA debe realizar, no texto libre
class RespuestaGuardian(BaseModel):

    answer: str = Field(
        description=(
            "Respuesta empática, cercana y humana para el usuario. "
            "Basada en la guía mhGAP de la OMS y la GPC de Conducta Suicida. "
            "Máximo 3-4 líneas. Tono de amigo cercano, nunca clínico ni frío. "
            "No saludes ni te despidas. Termina con una pregunta abierta si aplica."
        )
    )

    nivel_riesgo: Literal["ninguno", "leve", "moderado", "critico"] = Field(
        description=(
            "Nivel de riesgo detectado según la GPC de Conducta Suicida: "
            "'ninguno'  → sin señales de riesgo. "
            "'leve'     → ideación pasiva: 'ya no quiero estar aquí', cansancio vital, desesperanza general. "
            "'moderado' → ideación activa: piensa en hacerse daño pero sin plan claro. "
            "'critico'  → plan concreto, intención explícita, acceso a medios o intento previo reciente. "
            "En caso de duda entre dos niveles, elige siempre el mayor."
        )
    )

    riesgo_inminente: bool = Field(
        description=(
            "True ÚNICAMENTE si nivel_riesgo es 'critico'. "
            "Dispara el SMS al familiar y la alerta en el dashboard del psicólogo."
        )
    )

    sugerir_ejercicio: Optional[Literal["respiracion_478", "grounding_54321"]] = Field(
        default=None,
        description=(
            "Sugiere un ejercicio SOLO si la persona está muy ansiosa o abrumada: "
            "'respiracion_478'  → para ansiedad aguda, opresión en el pecho, pánico. "
            "'grounding_54321'  → para disociación, abrumamiento o desconexión de la realidad. "
            "En cualquier otro caso devuelve None."
        )
    )
# Cadena RAG
chatModel     = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
structured_llm = chatModel.with_structured_output(RespuestaGuardian)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
# Cadena rag completa
rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | structured_llm
)


# TIPOS DE AUDIO PERMITIDOS

TIPOS_AUDIO_PERMITIDOS = {'audio/webm', 'audio/wav', 'audio/ogg', 'audio/mpeg', 'audio/mp4'}

DICCIONARIO_EMOCIONES = {
    "neu": "neutral",
    "hap": "feliz",
    "ang": "enojado",
    "sad": "triste"
}


# FUNCIONES AUXILIARES 

def _detectar_emocion(ruta_archivo: str) -> str:
    if detector_emociones is None:
        return "neutral"
    try:
        audio_array, _ = librosa.load(ruta_archivo, sr=16000) # lee el arhivo de audio y lo convierte en un array de numero (16,000 Hz)
        resultado      = detector_emociones(audio_array) # pasa el array al modelo HuBERT para detectar la emocion del tono de voz, devuelve un label y score ejemplo :{'label': 'sad', 'score': 0.87}
        etiqueta       = resultado[0]['label']
        confianza      = resultado[0]['score']
        logger.info(f"Emoción detectada: {etiqueta} (confianza: {confianza:.2f})")
        return DICCIONARIO_EMOCIONES.get(etiqueta, "neutral") 
    except Exception as e:
        logger.error(f"Error en HuBERT: {e}")
        return "neutral"

# devuelve el texto transcrito del audio usando Whisper de OpenAI
def _transcribir_audio(ruta_archivo: str) -> str:
    try:
        with open(ruta_archivo, "rb") as f:
            transcripcion = client_openai.audio.transcriptions.create(
                model="whisper-1", file=f, language="es"
            )
        return transcripcion.text
    except Exception as e:
        logger.error(f"Error en Whisper: {e}")
        return ""


def _detectar_emocion_texto(texto: str) -> str:

    #Usa GPT-4o-mini para clasificar la emoción del texto

    try:
        respuesta = client_openai.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,          # determinista, no creativo
            max_tokens=10,          # solo necesita devolver 1 palabra
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres un clasificador de emociones. "
                        "Analiza el texto y responde ÚNICAMENTE con una de estas palabras, "
                        "sin puntuación ni explicación: "
                        "neutral, feliz, triste, enojado, ansioso"
                    )
                },
                {
                    "role": "user",
                    "content": texto
                }
            ]
        )

        emocion = respuesta.choices[0].message.content.strip().lower()

        # Validar que la respuesta sea una de las 5 categorías válidas
        EMOCIONES_VALIDAS = {"neutral", "feliz", "triste", "enojado", "ansioso"}
        if emocion not in EMOCIONES_VALIDAS:
            logger.warning(f"GPT devolvió emoción inesperada: '{emocion}' — usando 'neutral'")
            return "neutral"

        logger.info(f"Emoción en texto detectada: {emocion}")
        return emocion

    except Exception as e:
        logger.error(f"Error detectando emoción en texto: {e}")
        return "neutral"



# FUNCIONES DE BASE DE DATOS — historial y alertas

def _guardar_historial_emocional(id_usuario, emocion, fuente, riesgo, nivel_riesgo, mensaje, respuesta_ia):
    """
    Guarda cada interacción en historial_emocional.
    Se llama siempre al final de /get, sea audio o texto.
    """
    conexion = None
    cursor   = None
    try:
        conexion = ConexionDb.conexionBaseDeDatos()
        cursor   = conexion.cursor()
        sql = """
            INSERT INTO historial_emocional
                (id_usuario, emocion, fuente, riesgo_inminente, nivel_riesgo, mensaje_usuario, respuesta_ia)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(sql, (id_usuario, emocion, fuente, riesgo, nivel_riesgo,mensaje[:500] if mensaje else None,respuesta_ia[:1000] if respuesta_ia else None))
        conexion.commit()
        logger.info(f"Historial guardado — usuario {id_usuario}, emoción: {emocion}, riesgo: {riesgo}")
    except Exception as e:
        logger.error(f"Error guardando historial emocional: {e}")
    finally:
        if cursor:   cursor.close()
        if conexion: conexion.close()




def _registrar_alerta_crisis(id_usuario, mensaje_disparador, nivel="alto"):
    """
    Registra una alerta en alerta_crisis cuando riesgo_inminente=True.
    El psicólogo la verá destacada en el dashboard.
    """
    conexion = None
    cursor   = None
    try:
        conexion = ConexionDb.conexionBaseDeDatos()
        cursor   = conexion.cursor()
        sql = """
            INSERT INTO alerta_crisis (id_usuario, mensaje_disparador, nivel)
            VALUES (%s, %s, %s)
        """
        cursor.execute(sql, (id_usuario, mensaje_disparador[:500] if mensaje_disparador else "", nivel))
        conexion.commit()
        logger.warning(f"🚨 ALERTA DE CRISIS registrada — usuario {id_usuario}")
    except Exception as e:
        logger.error(f"Error registrando alerta de crisis: {e}")
    finally:
        if cursor:   cursor.close()
        if conexion: conexion.close()


def _enviar_sms_familiar(id_usuario: int, mensaje_crisis: str) -> bool:
    """
    Envía un SMS al familiar registrado del usuario cuando
    se detecta riesgo_inminente = True.
    Devuelve True si se envió correctamente.
    """
    conexion = None
    cursor   = None
    try:
        # 1. Obtener datos del usuario y su familiar
        conexion = ConexionDb.conexionBaseDeDatos()
        cursor   = conexion.cursor(dictionary=True)
        cursor.execute("""
            SELECT nombre, apellidos,
                   telefono_personal,
                   telefono_familiar,
                   nombre_familiar,
                   latitud_ultima,
                   longitud_ultima
            FROM usuario
            WHERE id_usuario = %s
        """, (id_usuario,))
        usuario = cursor.fetchone()

        if not usuario:
            logger.error(f"SMS: usuario {id_usuario} no encontrado")
            return False

        if not usuario.get('telefono_familiar'):
            logger.warning(f"SMS: usuario {id_usuario} no tiene teléfono familiar registrado")
            return False

        # 2. Construir el mensaje SMS
        nombre_usuario   = f"{usuario['nombre']} {usuario.get('apellidos', '')}".strip()
        nombre_familiar  = usuario.get('nombre_familiar', 'Familiar')
        tel_usuario      = usuario.get('telefono_personal', 'No registrado')
        lat              = usuario.get('latitud_ultima')
        lng              = usuario.get('longitud_ultima')

        # Enlace de ubicación Google Maps (si hay GPS)
        if lat and lng:
            link_mapa = f"https://www.google.com/maps?q={lat},{lng}"
            ubicacion_txt = f"📍 Ubicación aproximada: {link_mapa}"
        else:
            ubicacion_txt = "📍 Ubicación: no disponible"

        # Limitar el mensaje de crisis a 100 chars para no alargar el SMS
        fragmento = mensaje_crisis[:40] + ('...' if len(mensaje_crisis) > 40 else '')

# Ubicación en formato corto
        if lat and lng:
            ubicacion_txt = f"maps.google.com/?q={lat},{lng}"
        else:
            ubicacion_txt = "no disponible"

        cuerpo_sms = (
            f"GUARDIAN DIGITAL - ALERTA\n"
            f"{nombre_usuario} necesita ayuda.\n"
            f"Dijo: {fragmento}\n"
            f"Tel: {tel_usuario}\n"
            f"Ubic: {ubicacion_txt}"
        )

        # 3. Enviar SMS con Twilio
        cliente_twilio = TwilioClient(TWILIO_SID, TWILIO_TOKEN)
        mensaje = cliente_twilio.messages.create(
            body=cuerpo_sms,
            from_=TWILIO_NUMERO,
            to=f"+51{usuario['telefono_familiar']}"  # +51 = código de Perú
        )
        logger.warning(
            f"📱 SMS de crisis enviado al familiar de usuario {id_usuario} "
            f"— SID: {mensaje.sid}"
        )
        return True

    except Exception as e:
        logger.error(f"Error enviando SMS de crisis: {e}")
        return False
    finally:
        if cursor:   cursor.close()
        if conexion: conexion.close()



# DECORADOR: protege rutas que requieren login
def login_requerido(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'id_usuario' not in session:
            return redirect(url_for("home"))
        return f(*args, **kwargs)
    return decorated

def solo_profesional(f):
    """Bloquea acceso al dashboard si el usuario no es psicólogo."""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get('rol') != 'profesional':
            return redirect(url_for("chat_page"))
        return f(*args, **kwargs)
    return decorated


#  RUTAS PÚBLICAS
@app.route("/")
def home():
    return render_template("inicio.html")

@app.route("/inicio")
def inicio_page():
    return render_template("inicio.html")


# Ruta Inicio de sesión: verifica credenciales y redirige según rol 
@app.route("/login", methods=["POST"])
def login():

    correo      = request.form.get("txtCorreo")
    contrasenia = request.form.get("txtContrasenia")

    conexion = None
    cursor   = None

    try:
        conexion = ConexionDb.conexionBaseDeDatos()
        cursor   = conexion.cursor(dictionary=True)
        # Busca por correo al usuario que intenta loguearse
        cursor.execute("SELECT * FROM usuario WHERE correo = %s", (correo,))
        usuario  = cursor.fetchone()
    except Exception as e:
        logger.error(f"Error en login: {e}")
        return "Error en el servidor. Intenta de nuevo."
    finally:
        if cursor:   cursor.close()
        if conexion: conexion.close()
    # verifica que el usuario exista y que la contraseña coincida con el hash guardado en la base de datos
    if usuario and check_password_hash(usuario['contrasenia'], contrasenia):
        session['id_usuario'] = usuario['id_usuario']
        session['rol']        = usuario.get('rol', 'paciente')
        # Redireccion por rol
        if session['rol'] == 'profesional':
            return redirect(url_for("dashboard_profesional"))
        else:
            return redirect(url_for("chat_page"))
    else:
        return "Usuario o contraseña incorrectos"


# Ruta Cerrar sesión: limpia la sesión y redirige al inicio
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


# Ruta Chat: muestra la interfaz de chat con datos del usuario para personalizar la experiencia
@app.route('/chat')
@login_requerido
def chat_page():
    id_usuario_logueado = session['id_usuario']
    conexion = None
    cursor   = None
    try:
        conexion = ConexionDb.conexionBaseDeDatos()
        cursor   = conexion.cursor(dictionary=True)
        sql = """
            SELECT u.nombre, u.tema_color,
                   d.nombre AS distrito,
                   p.nombre AS provincia,
                   dep.nombre AS departamento
            FROM usuario u
            INNER JOIN distrito    d   ON u.id_distrito   = d.id_distrito
            INNER JOIN provincia   p   ON d.id_provincia  = p.id_provincia
            INNER JOIN departamento dep ON p.id_departamento = dep.id_departamento
            WHERE u.id_usuario = %s
        """
        cursor.execute(sql, (id_usuario_logueado,))
        datos_usuario = cursor.fetchone()
        # mostrar la plantilla de chat con los datos del usuario
        return render_template("chat.html", usuario=datos_usuario)
    except Exception as e:
        logger.error(f"Error cargando chat: {e}")
        return "Hubo un error al cargar tu perfil."
    finally:
        if cursor:   cursor.close()
        if conexion: conexion.close()

def _enviar_email_reset(correo_destino: str, nombre: str, token: str) -> bool:
    """
    Envía el correo con el enlace de recuperación de contraseña.
    Devuelve True si se envió correctamente.
    """
    try:
        enlace = f"{APP_URL}/reset/{token}"

        # ── Cuerpo HTML del correo ──────────────────────────
        html = f"""
        <div style="font-family:'Segoe UI',sans-serif;max-width:520px;margin:auto;
                    background:#0f1923;border-radius:16px;overflow:hidden;
                    border:1px solid rgba(255,255,255,.07);">

          <!-- Franja superior -->
          <div style="height:4px;background:linear-gradient(90deg,#4da3ff,#a78bfa,#0066ff)"></div>

          <!-- Cabecera -->
          <div style="padding:32px 36px 20px;text-align:center;">
            <div style="font-size:28px;margin-bottom:8px;">🛡️</div>
            <h2 style="color:#e8edf5;margin:0;font-size:20px;font-weight:700;">
              Guardian Digital
            </h2>
            <p style="color:#7a8fa8;font-size:13px;margin:4px 0 0;">
              Recuperación de contraseña
            </p>
          </div>

          <!-- Cuerpo -->
          <div style="padding:0 36px 32px;">
            <p style="color:#dde5f0;font-size:14px;line-height:1.6;">
              Hola <strong style="color:#4da3ff">{nombre}</strong>,
            </p>
            <p style="color:#8b95a8;font-size:14px;line-height:1.6;">
              Recibimos una solicitud para restablecer la contraseña de tu cuenta.
              Haz clic en el botón para crear una nueva contraseña:
            </p>

            <!-- Botón -->
            <div style="text-align:center;margin:28px 0;">
              <a href="{enlace}"
                 style="background:linear-gradient(135deg,#4da3ff,#0066ff);
                        color:#fff;text-decoration:none;padding:13px 32px;
                        border-radius:10px;font-weight:700;font-size:14px;
                        display:inline-block;">
                Restablecer contraseña
              </a>
            </div>

            <!-- Aviso expiración -->
            <div style="background:rgba(77,163,255,.06);border:1px solid rgba(77,163,255,.15);
                        border-left:3px solid #4da3ff;border-radius:8px;padding:12px 14px;">
              <p style="color:#7a8fa8;font-size:12px;margin:0;line-height:1.5;">
                ⏱️ Este enlace expira en <strong style="color:#4da3ff">30 minutos</strong>.<br>
                Si no solicitaste este cambio, ignora este correo. Tu contraseña no cambiará.
              </p>
            </div>

            <!-- Enlace alternativo -->
            <p style="color:#3d4f66;font-size:11px;margin-top:20px;word-break:break-all;">
              Si el botón no funciona, copia este enlace en tu navegador:<br>
              <span style="color:#4da3ff">{enlace}</span>
            </p>
          </div>

          <!-- Footer -->
          <div style="padding:16px 36px;border-top:1px solid rgba(255,255,255,.06);
                      text-align:center;">
            <p style="color:#3d4f66;font-size:11px;margin:0;">
              Guardian Digital · Sistema de apoyo emocional
            </p>
          </div>
        </div>
        """

        # ── Armar el mensaje MIME ───────────────────────────
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "🛡️ Guardian Digital — Recupera tu contraseña"
        msg["From"]    = f"Guardian Digital <{GMAIL_USER}>"
        msg["To"]      = correo_destino
        msg.attach(MIMEText(html, "html"))

        # ── Enviar via Gmail SMTP ───────────────────────────
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as servidor:
            servidor.login(GMAIL_USER, GMAIL_PASSWORD)
            servidor.sendmail(GMAIL_USER, correo_destino, msg.as_string())

        logger.info(f"✅ Email de recuperación enviado a {correo_destino}")
        return True

    except Exception as e:
        logger.error(f"Error enviando email de recuperación: {e}")
        return False


# RUTA GET: recibe mensajes del usuario (texto o audio), procesa la emoción, construye el prompt enriquecido y devuelve la respuesta de la IA
# CORAZON DEL SISTEMA
@app.route("/get", methods=["POST"])
@login_requerido
def get_response():
    try:
        msg           = ""
        es_audio      = False
        emocion_usuario = "neutral"

        # ----------------------------------------------------------
        # BLOQUE A: AUDIO — HuBERT + Whisper en PARALELO
        # ----------------------------------------------------------

        # Verifcamos si llego un archivo de audio
        if 'audio' in request.files:
            es_audio   = True
            audio_file = request.files['audio']

            if audio_file.content_type not in TIPOS_AUDIO_PERMITIDOS:
                logger.warning(f"Tipo de audio rechazado: {audio_file.content_type}")
                return jsonify({"answer": "Formato de audio no válido.", "riesgo_inminente": False})
            # Crear un archivo temporal para guardar el audio recibido y procesarlo con HuBERT y Whisper
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp:
                audio_file.save(tmp.name)
                ruta_temporal = tmp.name
            # Se utiliza los hilos simultaneos, HuBERT analiza la emocion del audio, Whisper transcribe el textto
            try:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    fut_emocion = executor.submit(_detectar_emocion, ruta_temporal)
                    fut_texto   = executor.submit(_transcribir_audio, ruta_temporal)
                    emocion_usuario = fut_emocion.result()
                    msg             = fut_texto.result()

                if not msg.strip():
                    msg = "No pude entender el audio con claridad."
                    logger.warning("Whisper devolvió texto vacío.")
            # Se ejecuta siempre eliminando el archivo temporal.
            finally:
                if os.path.exists(ruta_temporal):
                    os.remove(ruta_temporal)

        # ----------------------------------------------------------
        # BLOQUE B: TEXTO NORMAL
        # Si no hay audio lee el texto del chat
        else:
            msg = request.form.get("msg", "").strip()
            if not msg:
                return jsonify({
                    "answer": "No recibí ningún mensaje. ¿Puedes intentarlo de nuevo?",
                    "riesgo_inminente": False
                })

            # detectamos emoción del texto
            emocion_usuario = _detectar_emocion_texto(msg)

        # ----------------------------------------------------------
        # BLOQUE C: PERFIL DEL USUARIO para enriquecer el prompt
        # ----------------------------------------------------------
        id_usuario_logueado = session.get('id_usuario')
        perfil = None

        conexion = None
        cursor   = None
        try:
            conexion = ConexionDb.conexionBaseDeDatos()
            cursor   = conexion.cursor(dictionary=True)
            # Consultamos los datos de personalzacion del usuario para enriquecer el prompt
            cursor.execute(
                "SELECT edad, gustos, mascota_favorita, tono_lenguaje FROM usuario WHERE id_usuario = %s",
                (id_usuario_logueado,)
            )
            perfil = cursor.fetchone()
        except Exception as e:
            logger.error(f"Error consultando perfil: {e}")
        finally:
            if cursor:   cursor.close()
            if conexion: conexion.close()

        # ----------------------------------------------------------
        # BLOQUE D: CONSTRUCCIÓN DEL PROMPT ENRIQUECIDO
        # ----------------------------------------------------------
        if perfil:
            mensaje_enriquecido = f"""
            [CONTEXTO INTERNO — NO REVELAR AL USUARIO]

            Estás hablando con una persona real que necesita ser escuchada.
            Aquí tienes todo lo que sabes de ella para personalizar tu respuesta:

            PERFIL DE LA PERSONA:
            - Edad: {perfil['edad']} años.
            - Le gusta: {perfil['gustos']}.
            - Su compañero/a favorito: {perfil['mascota_favorita']} (menciónalo solo si surge natural, nunca forzado).
            - Cómo le gusta que le hablen: {perfil['tono_lenguaje']}.

            ESTADO EMOCIONAL DETECTADO AHORA: {emocion_usuario}

            CÓMO REACCIONAR SEGÚN SU EMOCIÓN:
            - Si suena TRISTE → Acompáñalo/a primero. Valida su dolor sin apresurarte a resolver.
              Si hay apertura, puedes mencionar algo de sus gustos ({perfil['gustos']}) para conectar.
            - Si suena ANSIOSO → Sé su ancla. Frases cortas, seguras, sin dramatismo.
              Puedes sugerir respiración si el momento lo pide.
            - Si suena ENOJADO → Ponte de su lado. Nunca digas 'cálmate'. Deja que se desahogue.
            - Si suena FELIZ → Acompáñalo/a en eso con energía genuina.
            - Si suena NEUTRAL → Charla natural, con su estilo preferido: {perfil['tono_lenguaje']}.

            REGLAS CRÍTICAS PARA ESTA RESPUESTA:
            1. NO saludes ni te despidas. La conversación ya empezó.
            2. Responde como un amigo cercano, NO como un bot ni un médico.
            3. Máximo 3-4 líneas. Conciso y cálido.
            4. Haz UNA sola pregunta abierta al final si tiene sentido.
            5. NUNCA menciones que detectaste su emoción ni que usas guías clínicas.
            6. Si hay señales de riesgo, activa el protocolo de seguridad del system prompt.

            LO QUE ACABA DE DECIRTE:
            "{msg}"
            """

        else:
            # Fallback si no se pudo cargar el perfil del usuario
            mensaje_enriquecido = f"""
            [CONTEXTO INTERNO — NO REVELAR AL USUARIO]

            Estado emocional detectado: {emocion_usuario}.

            REGLAS:
            - Responde como un amigo cercano, cálido y sin juicios.
            - NO saludes ni te despidas.
            - Máximo 3-4 líneas. Una sola pregunta abierta si aplica.
            - Si hay señales de riesgo, activa el protocolo de seguridad.
            - NUNCA reveles que eres IA ni que detectaste la emoción.

            Reacción según emoción:
            - TRISTE → valida y acompaña.
            - ANSIOSO → frases cortas y seguras.
            - ENOJADO → dale la razón, no digas 'cálmate'.
            - FELIZ → comparte su energía.
            - NEUTRAL → conversación relajada.

            LO QUE ACABA DE DECIRTE:
            "{msg}"
            """

        # ----------------------------------------------------------
        # BLOQUE E: RAG CHAIN → RESPUESTA DE LA IA
        # ----------------------------------------------------------
        response_obj = rag_chain.invoke(mensaje_enriquecido)

        # ----------------------------------------------------------
        # BLOQUE F: GUARDAR EN BASE DE DATOS ← NUEVO
        # Siempre guardamos el historial emocional de cada interacción
        # ----------------------------------------------------------
        fuente = "voz" if es_audio else "texto"
        _guardar_historial_emocional(
            id_usuario   = id_usuario_logueado,
            emocion      = emocion_usuario,
            fuente       = fuente,
            riesgo       = response_obj.riesgo_inminente,
            nivel_riesgo = response_obj.nivel_riesgo,
            mensaje      = msg,
            respuesta_ia = response_obj.answer
        )
        # si riesgo_inminente es True, registramos la alerta en la base de datos y enviamos un SMS al familiar del usuario
        if response_obj.nivel_riesgo != "ninguno":
            _registrar_alerta_crisis(
                id_usuario         = id_usuario_logueado,
                mensaje_disparador = msg,
                nivel              = response_obj.nivel_riesgo  # leve, moderado o critico
            )

        if response_obj.riesgo_inminente:  # solo critico dispara el SMS
            sms_enviado = _enviar_sms_familiar(
                id_usuario     = id_usuario_logueado,
                mensaje_crisis = msg
            )
            if sms_enviado:
                logger.warning(f"✅ SMS de crisis enviado para usuario {id_usuario_logueado}")
        # ----------------------------------------------------------
        # BLOQUE G: RESPUESTA AL FRONTEND
        # ----------------------------------------------------------
        return jsonify({
            "answer":           response_obj.answer,
            "riesgo_inminente": response_obj.riesgo_inminente,
            "sugerir_ejercicio": response_obj.sugerir_ejercicio,
            "texto_reconocido": msg if es_audio else None,
            "emocion_detectada": emocion_usuario if es_audio else None
        })

    except Exception as e:
        logger.error(f"Error general en /get: {e}", exc_info=True)
        return jsonify({"answer": "Tuve un pequeño problema técnico, ¿puedes repetirlo?", "riesgo_inminente": False})

# ============================================================
# DASHBOARD DEL PSICÓLOGO ← NUEVO
# ============================================================
@app.route('/dashboard')
@login_requerido
@solo_profesional
def dashboard_profesional():
    id_profesional = session['id_usuario']

    conexion = None
    cursor   = None
    try:
        conexion = ConexionDb.conexionBaseDeDatos()
        cursor   = conexion.cursor(dictionary=True)

        # Nombre del psicólogo para el header
        cursor.execute(
            "SELECT nombre, apellidos FROM usuario WHERE id_usuario = %s",
            (id_profesional,)
        )
        datos_profesional = cursor.fetchone()

        # Pacientes asignados — con departamento, teléfono y hora en UTC-5 (Perú)
        sql_pacientes = """
            SELECT
                u.id_usuario,
                u.nombre,
                u.apellidos,
                u.correo,
                u.edad,
                u.telefono_personal,
                dep.nombre          AS departamento,
                h.emocion           AS ultima_emocion,
                h.riesgo_inminente  AS ultimo_riesgo,
                h.nivel_riesgo      AS ultimo_nivel_riesgo,
                CONVERT_TZ(h.fecha, '+00:00', '-05:00') AS ultima_actividad
            FROM asignacion_paciente ap
            INNER JOIN usuario u ON ap.id_paciente = u.id_usuario
            LEFT JOIN distrito      d   ON u.id_distrito      = d.id_distrito
            LEFT JOIN provincia     p   ON d.id_provincia     = p.id_provincia
            LEFT JOIN departamento  dep ON p.id_departamento  = dep.id_departamento
            LEFT JOIN (
                SELECT id_usuario, emocion, riesgo_inminente, nivel_riesgo, fecha
                FROM historial_emocional h1
                WHERE fecha = (
                    SELECT MAX(fecha)
                    FROM historial_emocional h2
                    WHERE h2.id_usuario = h1.id_usuario
                )
            ) h ON u.id_usuario = h.id_usuario
            WHERE ap.id_profesional = %s AND ap.activo = TRUE
            ORDER BY
                CASE
                    WHEN h.nivel_riesgo = 'critico'  THEN 1
                    WHEN h.nivel_riesgo = 'moderado' THEN 2
                    WHEN h.nivel_riesgo = 'leve'     THEN 3
                    WHEN h.nivel_riesgo = 'ninguno'  THEN 4
                    ELSE 5  -- NULL / sin actividad → siempre al final
                END ASC,
                h.fecha DESC
        """
        cursor.execute(sql_pacientes, (id_profesional,))
        pacientes = cursor.fetchall()

        return render_template(
            "dashboard.html",
            profesional  = datos_profesional,
            pacientes    = pacientes,
            total_alertas = 0   # ya no se usa en el HTML, pero evita error de template
        )

    except Exception as e:
        logger.error(f"Error cargando dashboard: {e}")
        return "Hubo un error al cargar el dashboard."
    finally:
        if cursor:   cursor.close()
        if conexion: conexion.close()

# ============================================================
# HISTORIAL EMOCIONAL DE UN PACIENTE (para la gráfica)
# ============================================================
@app.route('/api/historial/<int:id_paciente>')
@login_requerido
@solo_profesional
def api_historial_paciente(id_paciente):
    id_profesional = session['id_usuario']

    conexion = None
    cursor   = None
    try:
        conexion = ConexionDb.conexionBaseDeDatos()
        cursor   = conexion.cursor(dictionary=True)
        # Comprobar que el paciente está asignado al profesional
        cursor.execute("""
            SELECT id_asignacion FROM asignacion_paciente
            WHERE id_profesional = %s AND id_paciente = %s AND activo = TRUE
        """, (id_profesional, id_paciente))

        if not cursor.fetchone():
            return jsonify({"error": "Paciente no autorizado"}), 403

        # Historial emocional con hora en UTC-5 (Perú)
        cursor.execute("""
            SELECT emocion, riesgo_inminente, fuente,
                   DATE_FORMAT(
                       CONVERT_TZ(fecha, '+00:00', '-05:00'),
                       '%d/%m %H:%i'
                   ) AS fecha_formateada
            FROM historial_emocional
            WHERE id_usuario = %s
            ORDER BY fecha DESC
            LIMIT 30
        """, (id_paciente,))
        historial = cursor.fetchall()

        # Conteo de emociones para gráfico de torta
        cursor.execute("""
            SELECT emocion, COUNT(*) AS total
            FROM historial_emocional
            WHERE id_usuario = %s
            GROUP BY emocion
        """, (id_paciente,))
        conteo_emociones = cursor.fetchall()

        # Alertas de crisis del paciente con hora en UTC-5 (Perú)
        cursor.execute("""
            SELECT
                nivel,
                mensaje_disparador,
                atendida,
                DATE_FORMAT(
                    CONVERT_TZ(fecha, '+00:00', '-05:00'),
                    '%d/%m/%Y %H:%i'
                ) AS fecha_formateada
            FROM alerta_crisis
            WHERE id_usuario = %s
            ORDER BY fecha DESC
            LIMIT 10
        """, (id_paciente,))
        alertas_paciente = cursor.fetchall()

        return jsonify({
            "historial":        list(reversed(historial)), # para la gráfica queremos el orden cronológico (más antiguo a más reciente)
            "conteo_emociones": conteo_emociones,
            "alertas_paciente": alertas_paciente
        })

    except Exception as e:
        logger.error(f"Error en api_historial_paciente: {e}")
        return jsonify({"error": "Error del servidor"}), 500
    finally:
        if cursor:   cursor.close()
        if conexion: conexion.close()

# ============================================================
# MARCAR ALERTA COMO ATENDIDA
# ============================================================
@app.route('/api/asignar_paciente', methods=["POST"])
@login_requerido
@solo_profesional
def asignar_paciente():
    id_paciente    = request.form.get("id_paciente", "").strip()
    id_profesional = session['id_usuario']

    conexion = None
    cursor   = None
    try:
        conexion = ConexionDb.conexionBaseDeDatos()
        cursor   = conexion.cursor(dictionary=True)

        # Verificar que el paciente exista y sea paciente
        cursor.execute(
            "SELECT id_usuario, nombre FROM usuario WHERE id_usuario = %s AND rol = 'paciente'",
            (id_paciente,)
        )
        paciente = cursor.fetchone()

        if not paciente:
            return jsonify({"success": False, "message": "Paciente no encontrado."})

        # Insertar asignación (INSERT IGNORE evita duplicados)
        cursor.execute("""
            INSERT IGNORE INTO asignacion_paciente (id_profesional, id_paciente)
            VALUES (%s, %s)
        """, (id_profesional, paciente['id_usuario']))
        conexion.commit()

        return jsonify({
            "success": True,
            "message": f"Paciente {paciente['nombre']} asignado correctamente."
        })

    except Exception as e:
        logger.error(f"Error asignando paciente: {e}")
        return jsonify({"success": False, "message": str(e)})
    finally:
        if cursor:   cursor.close()
        if conexion: conexion.close()


# ============================================================
# REGISTRO DE USUARIOS (pacientes)
# ============================================================
@app.route("/registroUsuario")
def registro():
    conexion = None
    cursor   = None
    try:
        conexion = ConexionDb.conexionBaseDeDatos()
        cursor   = conexion.cursor(dictionary=True)
        cursor.execute("SELECT * FROM departamento")
        departamentos = cursor.fetchall()
        return render_template("registroUsuario.html", departamentos=departamentos)
    except Exception as e:
        logger.error(f"Error cargando registro: {e}")
        return "Error cargando el formulario."
    finally:
        if cursor:   cursor.close()
        if conexion: conexion.close()

@app.route("/registrar", methods=["POST"])
def registrar_usuario():
    conexion = None
    cursor   = None
    try:
        conexion = ConexionDb.conexionBaseDeDatos()
        cursor   = conexion.cursor()

        nombre           = request.form["txtNombre"]
        apellidos        = request.form["txtApellidos"]
        contrasenia_hash = generate_password_hash(request.form["txtContrasenia"])
        correo           = request.form["txtCorreo"]
        edad             = request.form["txtEdad"]
        gusto            = request.form["txtgustos"]
        mascota          = request.form["txtmascota"]
        lenguaje         = request.form["txtlenguaje"]
        distrito         = request.form["selectDistrito"]
        tema_color       = request.form.get("txtTemaColor", "brisa_mar")
        telefono_personal  = request.form.get("txtTelefonoPersonal", "").strip()
        nombre_familiar    = request.form.get("txtNombreFamiliar", "").strip()
        telefono_familiar  = request.form.get("txtTelefonoFamiliar", "").strip()

        sql = """
            INSERT INTO usuario
            (nombre, apellidos, correo, contrasenia, edad, id_distrito,
            gustos, mascota_favorita, tono_lenguaje, tema_color, rol,
            telefono_personal, nombre_familiar, telefono_familiar)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'paciente',
            %s, %s, %s)
        """
        cursor.execute(sql, (
            nombre, apellidos, correo, contrasenia_hash,
            edad, distrito, gusto, mascota, lenguaje, tema_color,
            telefono_personal or None,
            nombre_familiar   or None,
            telefono_familiar or None
        ))
        conexion.commit()

        return jsonify({"success": True, "message": "Usuario registrado correctamente"})

    except Exception as e:
        import traceback
        logger.error(f"Error registrando usuario:\n{traceback.format_exc()}")
        return jsonify({"success": False, "message": str(e)})
    finally:
        if cursor:   cursor.close()
        if conexion: conexion.close()

# ============================================================
# CARGA DE PROVINCIAS Y DISTRITOS (formulario de registro)
# ============================================================
@app.route("/provincias/<int:id_departamento>")
def obtener_provincias(id_departamento):
    conexion = None
    cursor   = None
    try:
        conexion = ConexionDb.conexionBaseDeDatos()
        cursor   = conexion.cursor(dictionary=True)
        cursor.execute("SELECT id_provincia, nombre FROM provincia WHERE id_departamento = %s", (id_departamento,))
        return jsonify(cursor.fetchall())
    finally:
        if cursor:   cursor.close()
        if conexion: conexion.close()

@app.route("/distritos/<int:id_provincia>")
def obtener_distritos(id_provincia):
    conexion = None
    cursor   = None
    try:
        conexion = ConexionDb.conexionBaseDeDatos()
        cursor   = conexion.cursor(dictionary=True)
        cursor.execute("SELECT id_distrito, nombre FROM distrito WHERE id_provincia = %s", (id_provincia,))
        return jsonify(cursor.fetchall())
    finally:
        if cursor:   cursor.close()
        if conexion: conexion.close()

# ============================================================
# BOTIQUÍN DE CALMA
# ============================================================
@app.route('/botiquin')
@login_requerido
def botiquin_page():
    id_usuario_logueado = session['id_usuario']
    conexion = None
    cursor   = None
    try:
        conexion = ConexionDb.conexionBaseDeDatos()
        cursor   = conexion.cursor(dictionary=True)
        cursor.execute(
            "SELECT nombre, gustos, mascota_favorita FROM usuario WHERE id_usuario = %s",
            (id_usuario_logueado,)
        )
        datos_usuario = cursor.fetchone()
        return render_template("botiquin.html", usuario=datos_usuario)
    except Exception as e:
        logger.error(f"Error cargando botiquín: {e}")
        return "Hubo un error al cargar tu botiquín de calma."
    finally:
        if cursor:   cursor.close()
        if conexion: conexion.close()

# ============================================================
# ACTUALIZAR TEMA DE COLOR
# ============================================================
@app.route("/actualizar_tema", methods=["POST"])
@login_requerido
def actualizar_tema():
    nuevo_tema = request.form.get("tema")
    id_usuario = session['id_usuario']
    conexion = None
    cursor   = None
    try:
        conexion = ConexionDb.conexionBaseDeDatos()
        cursor   = conexion.cursor()
        cursor.execute("UPDATE usuario SET tema_color = %s WHERE id_usuario = %s", (nuevo_tema, id_usuario))
        conexion.commit()
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error actualizando tema: {e}")
        return jsonify({"success": False})
    finally:
        if cursor:   cursor.close()
        if conexion: conexion.close()

# ============================================================
# REGISTRO DE PSICÓLOGOS
# ============================================================
@app.route("/registro-profesional")
def registro_profesional():
    conexion = None
    cursor   = None
    try:
        conexion = ConexionDb.conexionBaseDeDatos()
        cursor   = conexion.cursor(dictionary=True)
        cursor.execute("SELECT * FROM departamento ORDER BY nombre")
        departamentos = cursor.fetchall()
        return render_template("registroProfesional.html", departamentos=departamentos)
    except Exception as e:
        logger.error(f"Error cargando registro profesional: {e}")
        return "Error cargando el formulario."
    finally:
        if cursor:   cursor.close()
        if conexion: conexion.close()


@app.route("/registrar-profesional", methods=["POST"])
def registrar_profesional():
    conexion = None
    cursor   = None
    try:
        nombre       = request.form["txtNombre"].strip()
        apellidos    = request.form["txtApellidos"].strip()
        correo       = request.form["txtCorreo"].strip()
        edad         = request.form["txtEdad"].strip()
        contrasenia  = request.form["txtContrasenia"]
        distrito     = request.form["selectDistrito"]
        cmp          = request.form["txtCMP"].strip()
        especialidad = request.form["txtEspecialidad"].strip()
        institucion  = request.form.get("txtInstitucion", "").strip()
        telefono     = request.form.get("txtTelefono", "").strip()

        if not all([nombre, apellidos, correo, edad, contrasenia, distrito, cmp, especialidad]):
            return jsonify({"success": False, "message": "Faltan campos obligatorios."})

        if len(contrasenia) < 8:
            return jsonify({"success": False, "message": "La contraseña debe tener al menos 8 caracteres."})

        contrasenia_hash = generate_password_hash(contrasenia)

        conexion = ConexionDb.conexionBaseDeDatos()
        cursor   = conexion.cursor(dictionary=True)

        # Verificar correo duplicado
        cursor.execute("SELECT id_usuario FROM usuario WHERE correo = %s", (correo,))
        if cursor.fetchone():
            return jsonify({"success": False, "message": "Ya existe una cuenta con ese correo."})

        # INSERT en usuario con rol='profesional'
        cursor.execute("""
            INSERT INTO usuario
                (nombre, apellidos, correo, contrasenia, edad, id_distrito, rol)
            VALUES (%s, %s, %s, %s, %s, %s, 'profesional')
        """, (nombre, apellidos, correo, contrasenia_hash, edad, distrito))
        id_nuevo_usuario = cursor.lastrowid

        # INSERT en perfil_profesional
        cursor.execute("""
            INSERT INTO perfil_profesional
                (id_usuario, cmp, especialidad, institucion, telefono_contacto)
            VALUES (%s, %s, %s, %s, %s)
        """, (id_nuevo_usuario, cmp, especialidad, institucion or None, telefono or None))

        conexion.commit()
        logger.info(f"✅ Nuevo psicólogo registrado: {correo} (id: {id_nuevo_usuario})")

        return jsonify({"success": True, "message": "Cuenta profesional creada correctamente."})

    except Exception as e:
        import traceback
        logger.error(f"Error registrando profesional:\n{traceback.format_exc()}")
        return jsonify({"success": False, "message": str(e)})
    finally:
        if cursor:   cursor.close()
        if conexion: conexion.close()

@app.route('/api/pacientes_disponibles')
@login_requerido
@solo_profesional
def pacientes_disponibles():
    conexion = None
    cursor   = None
    try:
        conexion = ConexionDb.conexionBaseDeDatos()
        cursor   = conexion.cursor(dictionary=True)

        # Pacientes que NO están en asignacion_paciente (activos)
        cursor.execute("""
            SELECT 
                u.id_usuario,
                u.nombre,
                u.apellidos,
                u.correo,
                u.edad,
                dep.nombre AS departamento
            FROM usuario u
            LEFT JOIN distrito    d   ON u.id_distrito    = d.id_distrito
            LEFT JOIN provincia   p   ON d.id_provincia   = p.id_provincia
            LEFT JOIN departamento dep ON p.id_departamento = dep.id_departamento
            WHERE u.rol = 'paciente'
              AND u.id_usuario NOT IN (
                  SELECT id_paciente FROM asignacion_paciente WHERE activo = TRUE
              )
            ORDER BY u.nombre ASC
        """)
        pacientes = cursor.fetchall()
        return jsonify({"pacientes": pacientes})

    except Exception as e:
        logger.error(f"Error obteniendo pacientes disponibles: {e}")
        return jsonify({"error": "Error del servidor"}), 500
    finally:
        if cursor:   cursor.close()
        if conexion: conexion.close()

@app.route('/actualizar_ubicacion', methods=['POST'])
@login_requerido
def actualizar_ubicacion():
    """
    El frontend del chat llama esta ruta con las coordenadas GPS
    cada vez que el usuario abre el chat.
    """
    lat = request.form.get('latitud')
    lng = request.form.get('longitud')
    id_usuario = session['id_usuario']

    if not lat or not lng:
        return jsonify({"success": False})

    conexion = None
    cursor   = None
    try:
        conexion = ConexionDb.conexionBaseDeDatos()
        cursor   = conexion.cursor()
        cursor.execute("""
            UPDATE usuario
            SET latitud_ultima = %s, longitud_ultima = %s
            WHERE id_usuario = %s
        """, (lat, lng, id_usuario))
        conexion.commit()
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error actualizando ubicación: {e}")
        return jsonify({"success": False})
    finally:
        if cursor:   cursor.close()
        if conexion: conexion.close()


# ============================================================
# RUTA 1 — GET muestra formulario / POST procesa solicitud
# ============================================================
@app.route("/recuperar-contrasenia", methods=["GET", "POST"])
def recuperar_contrasenia():
    if request.method == "GET":
        return render_template("recuperarContrasenia.html")

    # ── POST: procesar correo ──────────────────────────────
    correo = request.form.get("txtCorreo", "").strip().lower()

    if not correo:
        return jsonify({"success": False, "message": "Ingresa tu correo."})

    conexion = None
    cursor   = None
    try:
        conexion = ConexionDb.conexionBaseDeDatos()
        cursor   = conexion.cursor(dictionary=True)

        cursor.execute(
            "SELECT id_usuario, nombre FROM usuario WHERE correo = %s", (correo,)
        )
        usuario = cursor.fetchone()

        # Respuesta genérica por seguridad — no revelar si el correo existe
        if not usuario:
            return jsonify({
                "success": True,
                "message": "Si ese correo está registrado, recibirás un enlace en breve."
            })

        token  = secrets.token_urlsafe(48)
        expiry = datetime.now() + timedelta(minutes=30)

        cursor.execute("""
            UPDATE usuario
            SET reset_token = %s, reset_token_expiry = %s
            WHERE id_usuario = %s
        """, (token, expiry, usuario['id_usuario']))
        conexion.commit()

        _enviar_email_reset(correo, usuario['nombre'], token)

        return jsonify({
            "success": True,
            "message": "Si ese correo está registrado, recibirás un enlace en breve."
        })

    except Exception as e:
        logger.error(f"Error en recuperar_contrasenia: {e}")
        return jsonify({"success": False, "message": "Error del servidor. Intenta de nuevo."})
    finally:
        if cursor:   cursor.close()
        if conexion: conexion.close()


# ============================================================
# RUTA 2 — GET muestra form nueva contraseña / POST la guarda
# ============================================================
@app.route("/reset/<token>", methods=["GET", "POST"])
def reset_contrasenia(token):

    # ── GET: validar token y mostrar formulario ────────────
    if request.method == "GET":
        conexion = None
        cursor   = None
        try:
            conexion = ConexionDb.conexionBaseDeDatos()
            cursor   = conexion.cursor(dictionary=True)

            cursor.execute("""
                SELECT id_usuario, reset_token_expiry
                FROM usuario
                WHERE reset_token = %s
            """, (token,))
            usuario = cursor.fetchone()

            if not usuario:
                return render_template("resetContrasenia.html",
                                       token=None,
                                       error="El enlace no es válido.")

            if datetime.now() > usuario['reset_token_expiry']:
                return render_template("resetContrasenia.html",
                                       token=None,
                                       error="El enlace ha expirado. Solicita uno nuevo.")

            return render_template("resetContrasenia.html", token=token, error=None)

        except Exception as e:
            logger.error(f"Error en reset GET: {e}")
            return render_template("resetContrasenia.html",
                                   token=None, error="Error del servidor.")
        finally:
            if cursor:   cursor.close()
            if conexion: conexion.close()

    # ── POST: guardar nueva contraseña ─────────────────────
    nueva     = request.form.get("txtNuevaContrasenia", "").strip()
    confirmar = request.form.get("txtConfirmarContrasenia", "").strip()

    if not nueva or not confirmar:
        return jsonify({"success": False, "message": "Completa todos los campos."})

    if nueva != confirmar:
        return jsonify({"success": False, "message": "Las contraseñas no coinciden."})

    if len(nueva) < 8:
        return jsonify({"success": False, "message": "La contraseña debe tener al menos 8 caracteres."})

    conexion = None
    cursor   = None
    try:
        conexion = ConexionDb.conexionBaseDeDatos()
        cursor   = conexion.cursor(dictionary=True)

        cursor.execute("""
            SELECT id_usuario, reset_token_expiry
            FROM usuario
            WHERE reset_token = %s
        """, (token,))
        usuario = cursor.fetchone()

        if not usuario:
            return jsonify({"success": False, "message": "Enlace inválido."})

        if datetime.now() > usuario['reset_token_expiry']:
            return jsonify({"success": False, "message": "El enlace ha expirado."})

        nueva_hash = generate_password_hash(nueva)
        cursor.execute("""
            UPDATE usuario
            SET contrasenia        = %s,
                reset_token        = NULL,
                reset_token_expiry = NULL
            WHERE id_usuario = %s
        """, (nueva_hash, usuario['id_usuario']))
        conexion.commit()

        logger.info(f"✅ Contraseña actualizada para usuario {usuario['id_usuario']}")
        return jsonify({"success": True, "message": "Contraseña actualizada correctamente."})

    except Exception as e:
        logger.error(f"Error en reset POST: {e}")
        return jsonify({"success": False, "message": "Error del servidor."})
    finally:
        if cursor:   cursor.close()
        if conexion: conexion.close()

# ============================================================
# DIRECTORIO DE PROFESIONALES (para el paciente)
# ============================================================
@app.route('/api/profesionales')
@login_requerido
def api_profesionales():
    id_paciente = session['id_usuario']
    conexion = None
    cursor   = None
    try:
        conexion = ConexionDb.conexionBaseDeDatos()
        cursor   = conexion.cursor(dictionary=True)

        # Verificar si el paciente ya tiene profesional asignado
        cursor.execute("""
            SELECT id_asignacion FROM asignacion_paciente
            WHERE id_paciente = %s AND activo = TRUE
        """, (id_paciente,))
        ya_asignado = cursor.fetchone() is not None

        # Verificar si ya tiene solicitud pendiente
        cursor.execute("""
            SELECT id_solicitud, id_profesional FROM solicitud_apoyo
            WHERE id_paciente = %s AND estado = 'pendiente'
        """, (id_paciente,))
        solicitud = cursor.fetchone()

        # Lista de profesionales
        cursor.execute("""
            SELECT u.id_usuario, u.nombre, u.apellidos,
                   pp.especialidad, pp.institucion
            FROM usuario u
            INNER JOIN perfil_profesional pp ON u.id_usuario = pp.id_usuario
            WHERE u.rol = 'profesional'
            ORDER BY u.nombre ASC
        """)
        profesionales = cursor.fetchall()

        return jsonify({
            "profesionales":  profesionales,
            "ya_asignado":    ya_asignado,
            "id_solicitado":  solicitud['id_profesional'] if solicitud else None
        })
    except Exception as e:
        logger.error(f"Error obteniendo profesionales: {e}")
        return jsonify({"error": "Error del servidor"}), 500
    finally:
        if cursor:   cursor.close()
        if conexion: conexion.close()


# ============================================================
# SOLICITAR APOYO PROFESIONAL (paciente)
# ============================================================
@app.route('/api/solicitar_apoyo', methods=["POST"])
@login_requerido
def solicitar_apoyo():
    id_paciente    = session['id_usuario']
    id_profesional = request.form.get("id_profesional")
    conexion = None
    cursor   = None
    try:
        conexion = ConexionDb.conexionBaseDeDatos()
        cursor   = conexion.cursor(dictionary=True)

        # Verificar que no tenga ya una solicitud pendiente o asignación
        cursor.execute("""
            SELECT id_solicitud FROM solicitud_apoyo
            WHERE id_paciente = %s AND estado = 'pendiente'
        """, (id_paciente,))
        if cursor.fetchone():
            return jsonify({"success": False, "message": "Ya tienes una solicitud pendiente."})

        cursor.execute("""
            SELECT id_asignacion FROM asignacion_paciente
            WHERE id_paciente = %s AND activo = TRUE
        """, (id_paciente,))
        if cursor.fetchone():
            return jsonify({"success": False, "message": "Ya tienes un profesional asignado."})

        cursor.execute("""
            INSERT INTO solicitud_apoyo (id_paciente, id_profesional)
            VALUES (%s, %s)
        """, (id_paciente, id_profesional))
        conexion.commit()

        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error en solicitar_apoyo: {e}")
        return jsonify({"success": False, "message": "Error del servidor."}), 500
    finally:
        if cursor:   cursor.close()
        if conexion: conexion.close()


# ============================================================
# NOTIFICACIONES DEL PROFESIONAL (solicitudes pendientes)
# ============================================================
@app.route('/api/notificaciones')
@login_requerido
@solo_profesional
def api_notificaciones():
    id_profesional = session['id_usuario']
    conexion = None
    cursor   = None
    try:
        conexion = ConexionDb.conexionBaseDeDatos()
        cursor   = conexion.cursor(dictionary=True)
        cursor.execute("""
            SELECT sa.id_solicitud, sa.fecha,
                   u.nombre, u.apellidos, u.id_usuario AS id_paciente
            FROM solicitud_apoyo sa
            INNER JOIN usuario u ON sa.id_paciente = u.id_usuario
            WHERE sa.id_profesional = %s AND sa.estado = 'pendiente'
            ORDER BY sa.fecha DESC
        """, (id_profesional,))
        notificaciones = cursor.fetchall()

        # Formatear fecha
        for n in notificaciones:
            if n['fecha']:
                n['fecha'] = n['fecha'].strftime('%d/%m/%Y %H:%M')

        return jsonify({"notificaciones": notificaciones})
    except Exception as e:
        logger.error(f"Error obteniendo notificaciones: {e}")
        return jsonify({"error": "Error del servidor"}), 500
    finally:
        if cursor:   cursor.close()
        if conexion: conexion.close()


# ============================================================
# ATENDER SOLICITUD (profesional acepta y se asigna el paciente)
# ============================================================
@app.route('/api/atender_solicitud', methods=["POST"])
@login_requerido
@solo_profesional
def atender_solicitud():
    id_profesional = session['id_usuario']
    id_solicitud   = request.form.get("id_solicitud")
    conexion = None
    cursor   = None
    try:
        conexion = ConexionDb.conexionBaseDeDatos()
        cursor   = conexion.cursor(dictionary=True)

        # Obtener la solicitud
        cursor.execute("""
            SELECT id_paciente FROM solicitud_apoyo
            WHERE id_solicitud = %s AND id_profesional = %s AND estado = 'pendiente'
        """, (id_solicitud, id_profesional))
        solicitud = cursor.fetchone()

        if not solicitud:
            return jsonify({"success": False, "message": "Solicitud no encontrada."})

        id_paciente = solicitud['id_paciente']

        # Asignar paciente (INSERT IGNORE evita duplicados)
        cursor.execute("""
            INSERT IGNORE INTO asignacion_paciente (id_profesional, id_paciente)
            VALUES (%s, %s)
        """, (id_profesional, id_paciente))

        # Marcar solicitud como atendida
        cursor.execute("""
            UPDATE solicitud_apoyo SET estado = 'atendida'
            WHERE id_solicitud = %s
        """, (id_solicitud,))

        conexion.commit()
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error en atender_solicitud: {e}")
        return jsonify({"success": False, "message": "Error del servidor."}), 500
    finally:
        if cursor:   cursor.close()
        if conexion: conexion.close()
# ============================================================
# ARRANQUE
# ============================================================
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
