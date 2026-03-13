# Herramientas principales de Flask
from flask import Flask, render_template, jsonify, request, redirect, url_for, session
from src.helper import download_embeddings
from src.prompt import system_prompt
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from conexionDb.conexionDb import ConexionDb
from werkzeug.security import check_password_hash, generate_password_hash
from concurrent.futures import ThreadPoolExecutor
import tempfile
import logging
import os
import warnings
from openai import OpenAI
import librosa
from transformers import pipeline
from twilio.rest import Client as TwilioClient
from typing import Optional, Literal
import smtplib
import secrets
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta, timezone
from functools import wraps
import boto3
from botocore.exceptions import ClientError

# BLOQUE 2 — CONFIGURACIÓN S3
# Agregar junto a las demás variables de entorno (después de load_dotenv())
# ============================================================
 
AWS_ACCESS_KEY     = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY     = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION         = os.getenv("AWS_REGION", "us-east-2")
S3_BUCKET          = os.getenv("S3_BUCKET_NAME")
 
s3_client = boto3.client(
    "s3",
    region_name        = AWS_REGION,
    aws_access_key_id  = AWS_ACCESS_KEY,
    aws_secret_access_key = AWS_SECRET_KEY
)
 
TIPOS_DOCUMENTO_PERMITIDOS = {
    'application/pdf', 'image/jpeg', 'image/png', 'image/jpg'
}

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

detector_emociones = None
try:
    logger.info("Cargando modelo HuBERT de emociones...")
    detector_emociones = pipeline("audio-classification", model="superb/hubert-large-superb-er")
    logger.info("Modelo HuBERT listo.")
except Exception as e:
    logger.warning(f"Modelo HuBERT no disponible: {e}. La app seguira sin deteccion por voz.")

app = Flask(__name__)
load_dotenv()

TWILIO_SID    = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN  = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_NUMERO = os.getenv("TWILIO_PHONE_NUMBER")

app.secret_key                   = os.getenv("FLASK_SECRET_KEY")
PINECONE_API_KEY                 = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY                   = os.getenv("OPENAI_API_KEY")
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"]   = OPENAI_API_KEY

client_openai = OpenAI(api_key=OPENAI_API_KEY)

GMAIL_USER     = os.getenv("GMAIL_USER")
GMAIL_PASSWORD = os.getenv("GMAIL_PASSWORD")
APP_URL        = os.getenv("APP_URL", "http://localhost:8080")

embeddings = download_embeddings()
docsearch  = PineconeVectorStore.from_existing_index(
    index_name="guardian-digital",
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 2})


class RespuestaGuardian(BaseModel):
    answer: str = Field(
        description=(
            "Respuesta empatica, cercana y humana para el usuario. "
            "Basada en la guia mhGAP de la OMS y la GPC de Conducta Suicida. "
            "Maximo 3-4 lineas. Tono de amigo cercano, nunca clinico ni frio. "
            "No saludes ni te despidas. Termina con una pregunta abierta si aplica."
        )
    )
    nivel_riesgo: Literal["ninguno", "leve", "moderado", "critico"] = Field(
        description=(
            "Nivel de riesgo detectado segun la GPC de Conducta Suicida: "
            "'ninguno'  sin senales de riesgo. "
            "'leve'     ideacion pasiva. "
            "'moderado' ideacion activa sin plan claro. "
            "'critico'  plan concreto, intencion explicita o intento previo reciente. "
            "En caso de duda entre dos niveles, elige siempre el mayor."
        )
    )
    riesgo_inminente: bool = Field(
        description=(
            "True UNICAMENTE si nivel_riesgo es 'critico'. "
            "Dispara el SMS al familiar y la alerta en el dashboard."
        )
    )
    sugerir_ejercicio: Optional[Literal["respiracion_478", "grounding_54321"]] = Field(
        default=None,
        description=(
            "Sugiere un ejercicio SOLO si la persona esta muy ansiosa o abrumada. "
            "En cualquier otro caso devuelve None."
        )
    )


chatModel      = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
structured_llm = chatModel.with_structured_output(RespuestaGuardian)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | structured_llm
)

TIPOS_AUDIO_PERMITIDOS = {'audio/webm', 'audio/wav', 'audio/ogg', 'audio/mpeg', 'audio/mp4'}

DICCIONARIO_EMOCIONES = {
    "neu": "neutral",
    "hap": "feliz",
    "ang": "enojado",
    "sad": "triste"
}


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================



LIMA_TZ = timezone(timedelta(hours=-5))
 
def _subir_documento_s3(archivo, id_usuario: int, tipo: str) -> str | None:
    """
    Sube un archivo a S3 y retorna la URL pública.
    Ruta en S3: documentos_profesionales/{id_usuario}/{tipo}_{timestamp}.ext
    """
    try:
        extension  = archivo.filename.rsplit('.', 1)[-1].lower()
        timestamp  = datetime.now().strftime('%Y%m%d%H%M%S')
        nombre_s3  = f"documentos_profesionales/{id_usuario}/{tipo}_{timestamp}.{extension}"
 
        s3_client.upload_fileobj(
            archivo,
            S3_BUCKET,
            nombre_s3,
            ExtraArgs={"ContentType": archivo.content_type}
        )
 
        url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{nombre_s3}"
        logger.info(f"Documento subido a S3: {url}")
        return url, nombre_s3
 
    except ClientError as e:
        logger.error(f"Error subiendo a S3: {e}")
        return None, None

# BLOQUE 4 — FUNCIÓN AUXILIAR: email de resultado de validación
# Agregar en la sección de FUNCIONES AUXILIARES
# ============================================================
 
def _enviar_email_validacion(correo_destino: str, nombre: str, aprobado: bool, motivo: str = None) -> bool:
    try:
        if aprobado:
            asunto  = "Guardian Digital - Tu cuenta profesional fue aprobada"
            mensaje = f"""
            <p style="color:#dde5f0;font-size:14px;">
                Hola <strong style="color:#4da3ff">{nombre}</strong>,
            </p>
            <p style="color:#8b95a8;font-size:14px;line-height:1.6;">
                Tu postulación como profesional de salud mental en <strong>Guardian Digital</strong>
                ha sido <strong style="color:#22c55e">aprobada</strong>. ¡Ya puedes iniciar sesión!
            </p>
            <div style="text-align:center;margin:28px 0;">
                <a href="{APP_URL}"
                   style="background:linear-gradient(135deg,#4da3ff,#0066ff);
                          color:#fff;text-decoration:none;padding:13px 32px;
                          border-radius:10px;font-weight:700;font-size:14px;display:inline-block;">
                    Ingresar a Guardian Digital
                </a>
            </div>
            """
        else:
            asunto  = "Guardian Digital - Revisión de tu postulación profesional"
            mensaje = f"""
            <p style="color:#dde5f0;font-size:14px;">
                Hola <strong style="color:#4da3ff">{nombre}</strong>,
            </p>
            <p style="color:#8b95a8;font-size:14px;line-height:1.6;">
                Luego de revisar tu postulación, no pudimos aprobarla por el momento.
            </p>
            <div style="background:rgba(239,68,68,.08);border-left:3px solid #ef4444;
                        border-radius:8px;padding:12px 14px;margin:16px 0;">
                <p style="color:#f87171;font-size:13px;margin:0;line-height:1.5;">
                    <strong>Motivo:</strong> {motivo or 'No se especificó motivo.'}
                </p>
            </div>
            <p style="color:#8b95a8;font-size:13px;">
                Puedes volver a postular corrigiendo los documentos indicados.
            </p>
            """
 
        html = f"""
        <div style="font-family:'Segoe UI',sans-serif;max-width:520px;margin:auto;
                    background:#0f1923;border-radius:16px;overflow:hidden;
                    border:1px solid rgba(255,255,255,.07);">
          <div style="height:4px;background:linear-gradient(90deg,#4da3ff,#a78bfa,#0066ff)"></div>
          <div style="padding:32px 36px 20px;text-align:center;">
            <h2 style="color:#e8edf5;margin:0;font-size:20px;font-weight:700;">Guardian Digital</h2>
            <p style="color:#7a8fa8;font-size:13px;margin:4px 0 0;">Validación de cuenta profesional</p>
          </div>
          <div style="padding:0 36px 32px;">
            {mensaje}
          </div>
          <div style="padding:16px 36px;border-top:1px solid rgba(255,255,255,.06);text-align:center;">
            <p style="color:#3d4f66;font-size:11px;margin:0;">Guardian Digital - Sistema de apoyo emocional</p>
          </div>
        </div>
        """
 
        msg            = MIMEMultipart("alternative")
        msg["Subject"] = asunto
        msg["From"]    = f"Guardian Digital <{GMAIL_USER}>"
        msg["To"]      = correo_destino
        msg.attach(MIMEText(html, "html"))
 
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as servidor:
            servidor.login(GMAIL_USER, GMAIL_PASSWORD)
            servidor.sendmail(GMAIL_USER, correo_destino, msg.as_string())
 
        logger.info(f"Email de validación enviado a {correo_destino}")
        return True
 
    except Exception as e:
        logger.error(f"Error enviando email de validación: {e}")
        return False
 

def a_hora_lima(dt):
    """Convierte un datetime UTC de la BD a hora de Lima (UTC-5)."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(LIMA_TZ)


def _detectar_emocion(ruta_archivo: str) -> str:
    if detector_emociones is None:
        return "neutral"
    try:
        audio_array, _ = librosa.load(ruta_archivo, sr=16000)
        resultado      = detector_emociones(audio_array)
        etiqueta       = resultado[0]['label']
        confianza      = resultado[0]['score']
        logger.info(f"Emocion detectada: {etiqueta} (confianza: {confianza:.2f})")
        return DICCIONARIO_EMOCIONES.get(etiqueta, "neutral")
    except Exception as e:
        logger.error(f"Error en HuBERT: {e}")
        return "neutral"


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
    try:
        respuesta = client_openai.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=10,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres un clasificador de emociones. "
                        "Analiza el texto y responde UNICAMENTE con una de estas palabras, "
                        "sin puntuacion ni explicacion: "
                        "neutral, feliz, triste, enojado, ansioso"
                    )
                },
                {"role": "user", "content": texto}
            ]
        )
        emocion = respuesta.choices[0].message.content.strip().lower()
        EMOCIONES_VALIDAS = {"neutral", "feliz", "triste", "enojado", "ansioso"}
        if emocion not in EMOCIONES_VALIDAS:
            logger.warning(f"GPT devolvio emocion inesperada: '{emocion}' usando 'neutral'")
            return "neutral"
        logger.info(f"Emocion en texto detectada: {emocion}")
        return emocion
    except Exception as e:
        logger.error(f"Error detectando emocion en texto: {e}")
        return "neutral"


def _guardar_historial_emocional(id_usuario, emocion, fuente, riesgo, nivel_riesgo, mensaje, respuesta_ia):
    conexion = None
    cursor   = None
    try:
        conexion = ConexionDb.conexionBaseDeDatos()
        cursor   = conexion.cursor()
        cursor.execute("""
            INSERT INTO historial_emocional
                (id_usuario, emocion, fuente, riesgo_inminente, nivel_riesgo, mensaje_usuario, respuesta_ia)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            id_usuario, emocion, fuente, riesgo, nivel_riesgo,
            mensaje[:500]       if mensaje      else None,
            respuesta_ia[:1000] if respuesta_ia else None
        ))
        conexion.commit()
        logger.info(f"Historial guardado usuario {id_usuario}, emocion: {emocion}, riesgo: {riesgo}")
    except Exception as e:
        logger.error(f"Error guardando historial emocional: {e}")
    finally:
        if cursor:   cursor.close()
        if conexion: conexion.close()


def _registrar_alerta_crisis(id_usuario, mensaje_disparador, nivel="alto"):
    conexion = None
    cursor   = None
    try:
        conexion = ConexionDb.conexionBaseDeDatos()
        cursor   = conexion.cursor()
        cursor.execute("""
            INSERT INTO alerta_crisis (id_usuario, mensaje_disparador, nivel)
            VALUES (%s, %s, %s)
        """, (id_usuario, mensaje_disparador[:500] if mensaje_disparador else "", nivel))
        conexion.commit()
        logger.warning(f"ALERTA DE CRISIS registrada usuario {id_usuario}")
    except Exception as e:
        logger.error(f"Error registrando alerta de crisis: {e}")
    finally:
        if cursor:   cursor.close()
        if conexion: conexion.close()


def _enviar_sms_familiar(id_usuario: int, mensaje_crisis: str) -> bool:
    conexion = None
    cursor   = None
    try:
        conexion = ConexionDb.conexionBaseDeDatos()
        cursor   = conexion.cursor(dictionary=True)
        cursor.execute("""
            SELECT nombre, apellidos,
                   telefono_personal, telefono_familiar, nombre_familiar,
                   latitud_ultima, longitud_ultima
            FROM usuario WHERE id_usuario = %s
        """, (id_usuario,))
        usuario = cursor.fetchone()

        if not usuario:
            logger.error(f"SMS: usuario {id_usuario} no encontrado")
            return False
        if not usuario.get('telefono_familiar'):
            logger.warning(f"SMS: usuario {id_usuario} sin telefono familiar")
            return False

        nombre_usuario = f"{usuario['nombre']} {usuario.get('apellidos', '')}".strip()
        tel_usuario    = usuario.get('telefono_personal', 'No registrado')
        lat            = usuario.get('latitud_ultima')
        lng            = usuario.get('longitud_ultima')
        fragmento      = mensaje_crisis[:40] + ('...' if len(mensaje_crisis) > 40 else '')
        ubicacion_txt  = f"maps.google.com/?q={lat},{lng}" if (lat and lng) else "no disponible"

        cuerpo_sms = (
            f"GUARDIAN DIGITAL - ALERTA\n"
            f"{nombre_usuario} necesita ayuda.\n"
            f"Dijo: {fragmento}\n"
            f"Tel: {tel_usuario}\n"
            f"Ubic: {ubicacion_txt}"
        )

        cliente_twilio = TwilioClient(TWILIO_SID, TWILIO_TOKEN)
        mensaje = cliente_twilio.messages.create(
            body=cuerpo_sms,
            from_=TWILIO_NUMERO,
            to=f"+51{usuario['telefono_familiar']}"
        )
        logger.warning(f"SMS enviado al familiar de usuario {id_usuario} SID: {mensaje.sid}")
        return True

    except Exception as e:
        logger.error(f"Error enviando SMS: {e}")
        return False
    finally:
        if cursor:   cursor.close()
        if conexion: conexion.close()


def _enviar_email_reset(correo_destino: str, nombre: str, token: str) -> bool:
    try:
        enlace = f"{APP_URL}/reset/{token}"
        html = f"""
        <div style="font-family:'Segoe UI',sans-serif;max-width:520px;margin:auto;
                    background:#0f1923;border-radius:16px;overflow:hidden;
                    border:1px solid rgba(255,255,255,.07);">
          <div style="height:4px;background:linear-gradient(90deg,#4da3ff,#a78bfa,#0066ff)"></div>
          <div style="padding:32px 36px 20px;text-align:center;">
            <div style="font-size:28px;margin-bottom:8px;">shield</div>
            <h2 style="color:#e8edf5;margin:0;font-size:20px;font-weight:700;">Guardian Digital</h2>
            <p style="color:#7a8fa8;font-size:13px;margin:4px 0 0;">Recuperacion de contrasenia</p>
          </div>
          <div style="padding:0 36px 32px;">
            <p style="color:#dde5f0;font-size:14px;line-height:1.6;">
              Hola <strong style="color:#4da3ff">{nombre}</strong>,
            </p>
            <p style="color:#8b95a8;font-size:14px;line-height:1.6;">
              Recibimos una solicitud para restablecer tu contrasenia.
            </p>
            <div style="text-align:center;margin:28px 0;">
              <a href="{enlace}"
                 style="background:linear-gradient(135deg,#4da3ff,#0066ff);
                        color:#fff;text-decoration:none;padding:13px 32px;
                        border-radius:10px;font-weight:700;font-size:14px;display:inline-block;">
                Restablecer contrasenia
              </a>
            </div>
            <div style="background:rgba(77,163,255,.06);border-left:3px solid #4da3ff;
                        border-radius:8px;padding:12px 14px;">
              <p style="color:#7a8fa8;font-size:12px;margin:0;line-height:1.5;">
                Este enlace expira en 30 minutos. Si no lo solicitaste, ignora este correo.
              </p>
            </div>
            <p style="color:#3d4f66;font-size:11px;margin-top:20px;word-break:break-all;">
              Si el boton no funciona: <span style="color:#4da3ff">{enlace}</span>
            </p>
          </div>
          <div style="padding:16px 36px;border-top:1px solid rgba(255,255,255,.06);text-align:center;">
            <p style="color:#3d4f66;font-size:11px;margin:0;">Guardian Digital - Sistema de apoyo emocional</p>
          </div>
        </div>
        """
        msg            = MIMEMultipart("alternative")
        msg["Subject"] = "Guardian Digital - Recupera tu contrasenia"
        msg["From"]    = f"Guardian Digital <{GMAIL_USER}>"
        msg["To"]      = correo_destino
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as servidor:
            servidor.login(GMAIL_USER, GMAIL_PASSWORD)
            servidor.sendmail(GMAIL_USER, correo_destino, msg.as_string())

        logger.info(f"Email de recuperacion enviado a {correo_destino}")
        return True
    except Exception as e:
        logger.error(f"Error enviando email: {e}")
        return False


# ============================================================
# DECORADORES
# ============================================================

def login_requerido(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'id_usuario' not in session:
            return redirect(url_for("home"))
        return f(*args, **kwargs)
    return decorated


def solo_profesional(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get('rol') != 'profesional':
            return redirect(url_for("chat_page"))
        return f(*args, **kwargs)
    return decorated

def solo_admin(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get('rol') != 'admin':
            return redirect(url_for("chat_page"))
        return f(*args, **kwargs)
    return decorated


# ============================================================
# RUTAS PUBLICAS
# ============================================================

@app.route("/")
def home():
    return render_template("inicio.html")


@app.route("/inicio")
def inicio_page():
    return render_template("inicio.html")


@app.route("/login", methods=["POST"])
def login():
    correo      = request.form.get("txtCorreo")
    contrasenia = request.form.get("txtContrasenia")
    conexion = None
    cursor   = None
    try:
        conexion = ConexionDb.conexionBaseDeDatos()
        cursor   = conexion.cursor(dictionary=True)
        cursor.execute(
            "SELECT id_usuario, contrasenia, rol, estado_validacion FROM usuario WHERE correo = %s",
            (correo,)
        )
        usuario = cursor.fetchone()
    except Exception as e:
        logger.error(f"Error en login: {e}")
        return "Error en el servidor. Intenta de nuevo."
    finally:
        if cursor:   cursor.close()
        if conexion: conexion.close()
 
    if not usuario or not check_password_hash(usuario['contrasenia'], contrasenia):
        return "Usuario o contraseña incorrectos"
 
    rol = usuario.get('rol', 'paciente')
 
    # Bloqueo de profesionales no validados
    if rol == 'profesional':
        estado = usuario.get('estado_validacion', 'pendiente')
        if estado == 'pendiente':
            return "Tu cuenta está siendo revisada. Te notificaremos por correo cuando sea aprobada."
        if estado == 'rechazado':
            return "Tu postulación fue rechazada. Revisa tu correo para más detalles."
 
    session['id_usuario'] = usuario['id_usuario']
    session['rol']        = rol
 
    if rol == 'admin':
        return redirect(url_for("dashboard_admin"))
    elif rol == 'profesional':
        return redirect(url_for("dashboard_profesional"))
    else:
        return redirect(url_for("chat_page"))
 


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


# ============================================================
# CHAT
# ============================================================

@app.route('/chat')
@login_requerido
def chat_page():
    id_usuario_logueado = session['id_usuario']
    conexion = None
    cursor   = None
    try:
        conexion = ConexionDb.conexionBaseDeDatos()
        cursor   = conexion.cursor(dictionary=True)
        cursor.execute("""
            SELECT u.nombre, u.tema_color,
                   d.nombre   AS distrito,
                   p.nombre   AS provincia,
                   dep.nombre AS departamento
            FROM usuario u
            INNER JOIN distrito     d   ON u.id_distrito    = d.id_distrito
            INNER JOIN provincia    p   ON d.id_provincia   = p.id_provincia
            INNER JOIN departamento dep ON p.id_departamento = dep.id_departamento
            WHERE u.id_usuario = %s
        """, (id_usuario_logueado,))
        datos_usuario = cursor.fetchone()
        return render_template("chat.html", usuario=datos_usuario)
    except Exception as e:
        logger.error(f"Error cargando chat: {e}")
        return "Hubo un error al cargar tu perfil."
    finally:
        if cursor:   cursor.close()
        if conexion: conexion.close()


@app.route("/get", methods=["POST"])
@login_requerido
def get_response():
    try:
        msg             = ""
        es_audio        = False
        emocion_usuario = "neutral"

        if 'audio' in request.files:
            es_audio   = True
            audio_file = request.files['audio']

            if audio_file.content_type not in TIPOS_AUDIO_PERMITIDOS:
                logger.warning(f"Tipo de audio rechazado: {audio_file.content_type}")
                return jsonify({"answer": "Formato de audio no valido.", "riesgo_inminente": False})

            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp:
                audio_file.save(tmp.name)
                ruta_temporal = tmp.name

            try:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    fut_emocion     = executor.submit(_detectar_emocion, ruta_temporal)
                    fut_texto       = executor.submit(_transcribir_audio, ruta_temporal)
                    emocion_usuario = fut_emocion.result()
                    msg             = fut_texto.result()

                if not msg.strip():
                    msg = "No pude entender el audio con claridad."
                    logger.warning("Whisper devolvio texto vacio.")
            finally:
                if os.path.exists(ruta_temporal):
                    os.remove(ruta_temporal)
        else:
            msg = request.form.get("msg", "").strip()
            if not msg:
                return jsonify({
                    "answer": "No recibi ningun mensaje. Puedes intentarlo de nuevo?",
                    "riesgo_inminente": False
                })
            emocion_usuario = _detectar_emocion_texto(msg)

        id_usuario_logueado = session.get('id_usuario')
        perfil = None
        conexion = None
        cursor   = None
        try:
            conexion = ConexionDb.conexionBaseDeDatos()
            cursor   = conexion.cursor(dictionary=True)
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

        if perfil:
            mensaje_enriquecido = f"""
            [CONTEXTO INTERNO NO REVELAR AL USUARIO]
            Estas hablando con una persona real que necesita ser escuchada.

            PERFIL:
            - Edad: {perfil['edad']} anios.
            - Le gusta: {perfil['gustos']}.
            - Su companiero/a favorito: {perfil['mascota_favorita']}.
            - Como le gusta que le hablen: {perfil['tono_lenguaje']}.

            ESTADO EMOCIONAL DETECTADO: {emocion_usuario}

            REGLAS:
            1. NO saludes ni te despidas.
            2. Amigo cercano, NO bot ni medico.
            3. Maximo 3-4 lineas.
            4. UNA sola pregunta abierta al final si aplica.
            5. NUNCA menciones que detectaste emocion ni que usas guias clinicas.
            6. Si hay seniales de riesgo, activa el protocolo de seguridad.

            LO QUE ACABA DE DECIRTE:
            "{msg}"
            """
        else:
            mensaje_enriquecido = f"""
            [CONTEXTO INTERNO NO REVELAR AL USUARIO]
            Estado emocional detectado: {emocion_usuario}.
            REGLAS: amigo cercano, maximo 3-4 lineas, sin saludos ni despedidas.
            LO QUE ACABA DE DECIRTE: "{msg}"
            """

        response_obj = rag_chain.invoke(mensaje_enriquecido)

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

        if response_obj.nivel_riesgo != "ninguno":
            _registrar_alerta_crisis(
                id_usuario         = id_usuario_logueado,
                mensaje_disparador = msg,
                nivel              = response_obj.nivel_riesgo
            )

        if response_obj.riesgo_inminente:
            sms_enviado = _enviar_sms_familiar(id_usuario_logueado, msg)
            if sms_enviado:
                logger.warning(f"SMS enviado para usuario {id_usuario_logueado}")

        return jsonify({
            "answer":            response_obj.answer,
            "riesgo_inminente":  response_obj.riesgo_inminente,
            "sugerir_ejercicio": response_obj.sugerir_ejercicio,
            "texto_reconocido":  msg if es_audio else None,
            "emocion_detectada": emocion_usuario if es_audio else None
        })

    except Exception as e:
        logger.error(f"Error general en /get: {e}", exc_info=True)
        return jsonify({"answer": "Tuve un pequenio problema tecnico, puedes repetirlo?", "riesgo_inminente": False})


# ============================================================
# DASHBOARD DEL PSICOLOGO
# ============================================================

@app.route('/dashboard_profesional')
@login_requerido
@solo_profesional
def dashboard_profesional():
    id_profesional = session['id_usuario']
    conexion = None
    cursor   = None
    try:
        conexion = ConexionDb.conexionBaseDeDatos()
        cursor   = conexion.cursor(dictionary=True)

        cursor.execute(
            "SELECT nombre, apellidos FROM usuario WHERE id_usuario = %s", (id_profesional,)
        )
        datos_profesional = cursor.fetchone()

        # Incluye nombre_familiar y telefono_familiar para la nueva columna
        cursor.execute("""
            SELECT
                u.id_usuario,
                u.nombre,
                u.apellidos,
                u.correo,
                u.edad,
                u.telefono_personal,
                u.nombre_familiar,
                u.telefono_familiar,
                dep.nombre          AS departamento,
                h.emocion           AS ultima_emocion,
                h.riesgo_inminente  AS ultimo_riesgo,
                h.nivel_riesgo      AS ultimo_nivel_riesgo,
                CONVERT_TZ(h.fecha, '+00:00', '-05:00') AS ultima_actividad
            FROM asignacion_paciente ap
            INNER JOIN usuario u ON ap.id_paciente = u.id_usuario
            LEFT JOIN distrito      d   ON u.id_distrito     = d.id_distrito
            LEFT JOIN provincia     p   ON d.id_provincia    = p.id_provincia
            LEFT JOIN departamento  dep ON p.id_departamento = dep.id_departamento
            LEFT JOIN (
                SELECT id_usuario, emocion, riesgo_inminente, nivel_riesgo, fecha
                FROM historial_emocional h1
                WHERE fecha = (
                    SELECT MAX(fecha) FROM historial_emocional h2
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
                    ELSE 5
                END ASC,
                h.fecha DESC
        """, (id_profesional,))
        pacientes = cursor.fetchall()

        return render_template(
            "dashboard_profesional.html",
            profesional = datos_profesional,
            pacientes   = pacientes,
        )

    except Exception as e:
        logger.error(f"Error cargando dashboard: {e}")
        return "Hubo un error al cargar el dashboard."
    finally:
        if cursor:   cursor.close()
        if conexion: conexion.close()


# Ruta de compatibilidad con URL antigua
@app.route('/dashboard')
@login_requerido
@solo_profesional
def dashboard_redirect():
    return redirect(url_for("dashboard_profesional"))


# ============================================================
# HISTORIAL EMOCIONAL DE UN PACIENTE
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

        cursor.execute("""
            SELECT id_asignacion FROM asignacion_paciente
            WHERE id_profesional = %s AND id_paciente = %s AND activo = TRUE
        """, (id_profesional, id_paciente))
        if not cursor.fetchone():
            return jsonify({"error": "Paciente no autorizado"}), 403

        cursor.execute("""
            SELECT emocion, riesgo_inminente, fuente,
                   DATE_FORMAT(
                       CONVERT_TZ(fecha, '+00:00', '-05:00'), '%d/%m %H:%i'
                   ) AS fecha_formateada
            FROM historial_emocional
            WHERE id_usuario = %s
            ORDER BY fecha DESC LIMIT 30
        """, (id_paciente,))
        historial = cursor.fetchall()

        cursor.execute("""
            SELECT emocion, COUNT(*) AS total
            FROM historial_emocional
            WHERE id_usuario = %s
            GROUP BY emocion
        """, (id_paciente,))
        conteo_emociones = cursor.fetchall()

        cursor.execute("""
            SELECT nivel, mensaje_disparador, atendida,
                   DATE_FORMAT(
                       CONVERT_TZ(fecha, '+00:00', '-05:00'), '%d/%m/%Y %H:%i'
                   ) AS fecha_formateada
            FROM alerta_crisis
            WHERE id_usuario = %s
            ORDER BY fecha DESC LIMIT 10
        """, (id_paciente,))
        alertas_paciente = cursor.fetchall()

        return jsonify({
            "historial":        list(reversed(historial)),
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
# INTERVENCIONES CLINICAS
# ============================================================

@app.route('/historial_intervenciones')
@login_requerido
@solo_profesional
def historial_intervenciones():
    id_profesional = session['id_usuario']
    conexion = None
    cursor   = None
    try:
        conexion = ConexionDb.conexionBaseDeDatos()
        cursor   = conexion.cursor(dictionary=True)

        cursor.execute(
            "SELECT id_usuario, nombre, apellidos FROM usuario WHERE id_usuario = %s",
            (id_profesional,)
        )
        profesional = cursor.fetchone()

        cursor.execute("""
            SELECT u.id_usuario, u.nombre, u.apellidos
            FROM   asignacion_paciente ap
            JOIN   usuario u ON u.id_usuario = ap.id_paciente
            WHERE  ap.id_profesional = %s AND ap.activo = TRUE
            ORDER  BY u.nombre, u.apellidos
        """, (id_profesional,))
        pacientes = cursor.fetchall()

        return render_template(
            "historial_intervenciones.html",
            profesional = profesional,
            pacientes   = pacientes
        )

    except Exception as e:
        logger.error(f"Error cargando historial intervenciones: {e}")
        return "Error al cargar el historial de intervenciones."
    finally:
        if cursor:   cursor.close()
        if conexion: conexion.close()


@app.route('/api/intervenciones', methods=['POST'])
@login_requerido
@solo_profesional
def registrar_intervencion():
    id_profesional     = session['id_usuario']
    id_paciente        = request.form.get('id_paciente')
    fecha_intervencion = request.form.get('fecha_intervencion')
    duracion_minutos   = request.form.get('duracion_minutos') or None
    tipo               = request.form.get('tipo')
    estado_paciente    = request.form.get('estado_paciente')
    nota_clinica       = request.form.get('nota_clinica')  or None
    proxima_cita       = request.form.get('proxima_cita')  or None
    derivado_a         = request.form.get('derivado_a')    or None

    if not all([id_paciente, fecha_intervencion, tipo, estado_paciente]):
        return jsonify({'success': False, 'message': 'Faltan campos obligatorios'}), 400

    conexion = None
    cursor   = None
    try:
        conexion = ConexionDb.conexionBaseDeDatos()
        cursor   = conexion.cursor(dictionary=True)

        cursor.execute("""
            SELECT id_asignacion FROM asignacion_paciente
            WHERE id_profesional = %s AND id_paciente = %s AND activo = TRUE
        """, (id_profesional, id_paciente))
        if not cursor.fetchone():
            return jsonify({'success': False, 'message': 'Paciente no asignado'}), 403

        cursor.execute("""
            INSERT INTO intervencion
                (id_profesional, id_paciente, fecha_intervencion,
                 duracion_minutos, tipo, estado_paciente,
                 nota_clinica, proxima_cita, derivado_a)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            id_profesional, id_paciente, fecha_intervencion,
            duracion_minutos, tipo, estado_paciente,
            nota_clinica, proxima_cita, derivado_a
        ))
        conexion.commit()
        id_nueva = cursor.lastrowid
        logger.info(f"Intervencion {id_nueva} registrada profesional {id_profesional} paciente {id_paciente}")
        return jsonify({'success': True, 'id_intervencion': id_nueva})

    except Exception as e:
        logger.error(f"Error registrando intervencion: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500
    finally:
        if cursor:   cursor.close()
        if conexion: conexion.close()


@app.route('/api/intervenciones/<int:id_paciente>')
@login_requerido
@solo_profesional
def obtener_intervenciones_paciente(id_paciente):
    id_profesional = session['id_usuario']
    conexion = None
    cursor   = None
    try:
        conexion = ConexionDb.conexionBaseDeDatos()
        cursor   = conexion.cursor(dictionary=True)

        cursor.execute("""
            SELECT id_intervencion, fecha_intervencion, duracion_minutos,
                   tipo, estado_paciente, nota_clinica,
                   proxima_cita, derivado_a, fecha_registro
            FROM intervencion
            WHERE id_profesional = %s AND id_paciente = %s
            ORDER BY fecha_intervencion DESC
        """, (id_profesional, id_paciente))
        intervenciones = cursor.fetchall()

        resultado = []
        for inv in intervenciones:
            resultado.append({
                'id_intervencion':    inv['id_intervencion'],
                'fecha_intervencion': a_hora_lima(inv['fecha_intervencion']).strftime('%d/%m/%Y %H:%M'),
                'duracion_minutos':   inv['duracion_minutos'],
                'tipo':               inv['tipo'],
                'estado_paciente':    inv['estado_paciente'],
                'nota_clinica':       inv['nota_clinica'],
                'proxima_cita':       inv['proxima_cita'].strftime('%d/%m/%Y') if inv['proxima_cita'] else None,
                'derivado_a':         inv['derivado_a'],
                'fecha_registro':     a_hora_lima(inv['fecha_registro']).strftime('%d/%m/%Y %H:%M'),
            })

        return jsonify({'intervenciones': resultado})

    except Exception as e:
        logger.error(f"Error obteniendo intervenciones del paciente: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        if cursor:   cursor.close()
        if conexion: conexion.close()


@app.route('/api/todas_intervenciones')
@login_requerido
@solo_profesional
def todas_intervenciones():
    id_profesional     = session['id_usuario']
    id_paciente_filtro = request.args.get('id_paciente')
    conexion = None
    cursor   = None
    try:
        conexion = ConexionDb.conexionBaseDeDatos()
        cursor   = conexion.cursor(dictionary=True)

        query = """
            SELECT i.id_intervencion, i.fecha_intervencion, i.duracion_minutos,
                   i.tipo, i.estado_paciente, i.nota_clinica,
                   i.proxima_cita, i.derivado_a,
                   u.nombre    AS paciente_nombre,
                   u.apellidos AS paciente_apellidos,
                   u.correo    AS paciente_correo
            FROM  intervencion i
            JOIN  usuario u ON u.id_usuario = i.id_paciente
            WHERE i.id_profesional = %s
        """
        params = [id_profesional]

        if id_paciente_filtro:
            query  += " AND i.id_paciente = %s"
            params.append(id_paciente_filtro)

        query += " ORDER BY i.fecha_intervencion DESC"

        cursor.execute(query, params)
        intervenciones = cursor.fetchall()

        resultado = []
        for inv in intervenciones:
            resultado.append({
                'id_intervencion':    inv['id_intervencion'],
                'fecha_intervencion': a_hora_lima(inv['fecha_intervencion']).strftime('%d/%m/%Y %H:%M'),
                'duracion_minutos':   inv['duracion_minutos'],
                'tipo':               inv['tipo'],
                'estado_paciente':    inv['estado_paciente'],
                'nota_clinica':       inv['nota_clinica'],
                'proxima_cita':       inv['proxima_cita'].strftime('%d/%m/%Y') if inv['proxima_cita'] else None,
                'derivado_a':         inv['derivado_a'],
                'paciente_nombre':    inv['paciente_nombre'],
                'paciente_apellidos': inv['paciente_apellidos'],
                'paciente_correo':    inv['paciente_correo'],
            })

        return jsonify({'intervenciones': resultado, 'total': len(resultado)})

    except Exception as e:
        logger.error(f"Error en todas_intervenciones: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        if cursor:   cursor.close()
        if conexion: conexion.close()


# ============================================================
# ASIGNAR PACIENTE A PROFESIONAL
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
        cursor.execute(
            "SELECT id_usuario, nombre FROM usuario WHERE id_usuario = %s AND rol = 'paciente'",
            (id_paciente,)
        )
        paciente = cursor.fetchone()
        if not paciente:
            return jsonify({"success": False, "message": "Paciente no encontrado."})

        cursor.execute("""
            INSERT IGNORE INTO asignacion_paciente (id_profesional, id_paciente)
            VALUES (%s, %s)
        """, (id_profesional, paciente['id_usuario']))
        conexion.commit()
        return jsonify({"success": True, "message": f"Paciente {paciente['nombre']} asignado."})

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
        nombre             = request.form["txtNombre"]
        apellidos          = request.form["txtApellidos"]
        contrasenia_hash   = generate_password_hash(request.form["txtContrasenia"])
        correo             = request.form["txtCorreo"]
        edad               = request.form["txtEdad"]
        gusto              = request.form["txtgustos"]
        mascota            = request.form["txtmascota"]
        lenguaje           = request.form["txtlenguaje"]
        distrito           = request.form["selectDistrito"]
        tema_color         = request.form.get("txtTemaColor", "brisa_mar")
        telefono_personal  = request.form.get("txtTelefonoPersonal", "").strip()
        nombre_familiar    = request.form.get("txtNombreFamiliar",   "").strip()
        telefono_familiar  = request.form.get("txtTelefonoFamiliar", "").strip()

        cursor.execute("""
            INSERT INTO usuario
            (nombre, apellidos, correo, contrasenia, edad, id_distrito,
             gustos, mascota_favorita, tono_lenguaje, tema_color, rol,
             telefono_personal, nombre_familiar, telefono_familiar)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'paciente', %s, %s, %s)
        """, (
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
# PROVINCIAS Y DISTRITOS
# ============================================================

@app.route("/provincias/<int:id_departamento>")
def obtener_provincias(id_departamento):
    conexion = None
    cursor   = None
    try:
        conexion = ConexionDb.conexionBaseDeDatos()
        cursor   = conexion.cursor(dictionary=True)
        cursor.execute(
            "SELECT id_provincia, nombre FROM provincia WHERE id_departamento = %s",
            (id_departamento,)
        )
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
        cursor.execute(
            "SELECT id_distrito, nombre FROM distrito WHERE id_provincia = %s",
            (id_provincia,)
        )
        return jsonify(cursor.fetchall())
    finally:
        if cursor:   cursor.close()
        if conexion: conexion.close()


# ============================================================
# BOTIQUIN DE CALMA
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
        logger.error(f"Error cargando botiquin: {e}")
        return "Hubo un error al cargar tu botiquin de calma."
    finally:
        if cursor:   cursor.close()
        if conexion: conexion.close()


# ============================================================
# ACTUALIZAR TEMA
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
        cursor.execute(
            "UPDATE usuario SET tema_color = %s WHERE id_usuario = %s",
            (nuevo_tema, id_usuario)
        )
        conexion.commit()
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error actualizando tema: {e}")
        return jsonify({"success": False})
    finally:
        if cursor:   cursor.close()
        if conexion: conexion.close()


# ============================================================
# REGISTRO DE PSICOLOGOS
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
        telefono     = request.form.get("txtTelefono",    "").strip()
 
        if not all([nombre, apellidos, correo, edad, contrasenia, distrito, cmp, especialidad]):
            return jsonify({"success": False, "message": "Faltan campos obligatorios."})
        if len(contrasenia) < 8:
            return jsonify({"success": False, "message": "La contraseña debe tener al menos 8 caracteres."})
 
        # Validar que vienen los 3 archivos
        archivo_titulo  = request.files.get("docTitulo")
        archivo_cmp     = request.files.get("docCMP")
        archivo_espec   = request.files.get("docEspecializacion")
 
        if not archivo_titulo or not archivo_cmp:
            return jsonify({"success": False, "message": "Debes subir el título profesional y el certificado CMP."})
 
        # Validar tipos de archivo
        for archivo in [archivo_titulo, archivo_cmp]:
            if archivo.content_type not in TIPOS_DOCUMENTO_PERMITIDOS:
                return jsonify({"success": False, "message": f"Formato no permitido en {archivo.filename}. Usa PDF, JPG o PNG."})
 
        if archivo_espec and archivo_espec.filename:
            if archivo_espec.content_type not in TIPOS_DOCUMENTO_PERMITIDOS:
                return jsonify({"success": False, "message": f"Formato no permitido en {archivo_espec.filename}."})
 
        contrasenia_hash = generate_password_hash(contrasenia)
 
        conexion = ConexionDb.conexionBaseDeDatos()
        cursor   = conexion.cursor(dictionary=True)
 
        cursor.execute("SELECT id_usuario FROM usuario WHERE correo = %s", (correo,))
        if cursor.fetchone():
            return jsonify({"success": False, "message": "Ya existe una cuenta con ese correo."})
 
        # Crear usuario con estado pendiente
        cursor.execute("""
            INSERT INTO usuario
                (nombre, apellidos, correo, contrasenia, edad, id_distrito, rol, estado_validacion)
            VALUES (%s, %s, %s, %s, %s, %s, 'profesional', 'pendiente')
        """, (nombre, apellidos, correo, contrasenia_hash, edad, distrito))
        id_nuevo_usuario = cursor.lastrowid
 
        # Crear perfil profesional
        cursor.execute("""
            INSERT INTO perfil_profesional
                (id_usuario, cmp, especialidad, institucion, telefono_contacto)
            VALUES (%s, %s, %s, %s, %s)
        """, (id_nuevo_usuario, cmp, especialidad, institucion or None, telefono or None))
 
        conexion.commit()
 
        # Subir documentos a S3 y registrar en BD
        documentos = [
            (archivo_titulo, "titulo"),
            (archivo_cmp,    "cmp"),
        ]
        if archivo_espec and archivo_espec.filename:
            documentos.append((archivo_espec, "especializacion"))
 
        for archivo, tipo in documentos:
            url, nombre_s3 = _subir_documento_s3(archivo, id_nuevo_usuario, tipo)
            if url:
                cursor.execute("""
                    INSERT INTO documento_postulacion
                        (id_usuario, tipo, nombre_archivo, url_s3)
                    VALUES (%s, %s, %s, %s)
                """, (id_nuevo_usuario, tipo, archivo.filename, url))
            else:
                logger.warning(f"No se pudo subir documento tipo {tipo} para usuario {id_nuevo_usuario}")
 
        conexion.commit()
        logger.info(f"Nueva postulación profesional: {correo} (id: {id_nuevo_usuario})")
        return jsonify({
            "success": True,
            "message": "Postulación enviada. Revisaremos tus documentos y te notificaremos por correo."
        })
 
    except Exception as e:
        import traceback
        logger.error(f"Error registrando profesional:\n{traceback.format_exc()}")
        return jsonify({"success": False, "message": str(e)})
    finally:
        if cursor:   cursor.close()
        if conexion: conexion.close()


# ============================================================
# PACIENTES DISPONIBLES (sin asignar)
# ============================================================

@app.route('/api/pacientes_disponibles')
@login_requerido
@solo_profesional
def pacientes_disponibles():
    conexion = None
    cursor   = None
    try:
        conexion = ConexionDb.conexionBaseDeDatos()
        cursor   = conexion.cursor(dictionary=True)
        cursor.execute("""
            SELECT u.id_usuario, u.nombre, u.apellidos, u.correo, u.edad,
                   dep.nombre AS departamento
            FROM usuario u
            LEFT JOIN distrito      d   ON u.id_distrito    = d.id_distrito
            LEFT JOIN provincia     p   ON d.id_provincia   = p.id_provincia
            LEFT JOIN departamento  dep ON p.id_departamento = dep.id_departamento
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


# ============================================================
# ACTUALIZAR UBICACION GPS
# ============================================================

@app.route('/actualizar_ubicacion', methods=['POST'])
@login_requerido
def actualizar_ubicacion():
    lat        = request.form.get('latitud')
    lng        = request.form.get('longitud')
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
        logger.error(f"Error actualizando ubicacion: {e}")
        return jsonify({"success": False})
    finally:
        if cursor:   cursor.close()
        if conexion: conexion.close()


# ============================================================
# RECUPERACION DE CONTRASENIA
# ============================================================

@app.route("/recuperar-contrasenia", methods=["GET", "POST"])
def recuperar_contrasenia():
    if request.method == "GET":
        return render_template("recuperarContrasenia.html")

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

        if not usuario:
            return jsonify({"success": True, "message": "Si ese correo esta registrado, recibiras un enlace en breve."})

        token  = secrets.token_urlsafe(48)
        expiry = datetime.now() + timedelta(minutes=30)
        cursor.execute("""
            UPDATE usuario SET reset_token = %s, reset_token_expiry = %s
            WHERE id_usuario = %s
        """, (token, expiry, usuario['id_usuario']))
        conexion.commit()

        _enviar_email_reset(correo, usuario['nombre'], token)
        return jsonify({"success": True, "message": "Si ese correo esta registrado, recibiras un enlace en breve."})

    except Exception as e:
        logger.error(f"Error en recuperar_contrasenia: {e}")
        return jsonify({"success": False, "message": "Error del servidor."})
    finally:
        if cursor:   cursor.close()
        if conexion: conexion.close()


@app.route("/reset/<token>", methods=["GET", "POST"])
def reset_contrasenia(token):
    if request.method == "GET":
        conexion = None
        cursor   = None
        try:
            conexion = ConexionDb.conexionBaseDeDatos()
            cursor   = conexion.cursor(dictionary=True)
            cursor.execute(
                "SELECT id_usuario, reset_token_expiry FROM usuario WHERE reset_token = %s", (token,)
            )
            usuario = cursor.fetchone()
            if not usuario:
                return render_template("resetContrasenia.html", token=None, error="El enlace no es valido.")
            if datetime.now() > usuario['reset_token_expiry']:
                return render_template("resetContrasenia.html", token=None, error="El enlace ha expirado.")
            return render_template("resetContrasenia.html", token=token, error=None)
        except Exception as e:
            logger.error(f"Error en reset GET: {e}")
            return render_template("resetContrasenia.html", token=None, error="Error del servidor.")
        finally:
            if cursor:   cursor.close()
            if conexion: conexion.close()

    nueva     = request.form.get("txtNuevaContrasenia",    "").strip()
    confirmar = request.form.get("txtConfirmarContrasenia","").strip()

    if not nueva or not confirmar:
        return jsonify({"success": False, "message": "Completa todos los campos."})
    if nueva != confirmar:
        return jsonify({"success": False, "message": "Las contrasenias no coinciden."})
    if len(nueva) < 8:
        return jsonify({"success": False, "message": "La contrasenia debe tener al menos 8 caracteres."})

    conexion = None
    cursor   = None
    try:
        conexion = ConexionDb.conexionBaseDeDatos()
        cursor   = conexion.cursor(dictionary=True)
        cursor.execute(
            "SELECT id_usuario, reset_token_expiry FROM usuario WHERE reset_token = %s", (token,)
        )
        usuario = cursor.fetchone()
        if not usuario:
            return jsonify({"success": False, "message": "Enlace invalido."})
        if datetime.now() > usuario['reset_token_expiry']:
            return jsonify({"success": False, "message": "El enlace ha expirado."})

        cursor.execute("""
            UPDATE usuario
            SET contrasenia = %s, reset_token = NULL, reset_token_expiry = NULL
            WHERE id_usuario = %s
        """, (generate_password_hash(nueva), usuario['id_usuario']))
        conexion.commit()
        logger.info(f"Contrasenia actualizada para usuario {usuario['id_usuario']}")
        return jsonify({"success": True, "message": "Contrasenia actualizada correctamente."})

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

        cursor.execute("""
            SELECT id_asignacion FROM asignacion_paciente
            WHERE id_paciente = %s AND activo = TRUE
        """, (id_paciente,))
        ya_asignado = cursor.fetchone() is not None

        cursor.execute("""
            SELECT id_solicitud, id_profesional FROM solicitud_apoyo
            WHERE id_paciente = %s AND estado = 'pendiente'
        """, (id_paciente,))
        solicitud = cursor.fetchone()

        cursor.execute("""
            SELECT u.id_usuario, u.nombre, u.apellidos,
                pp.especialidad, pp.institucion
            FROM usuario u
            INNER JOIN perfil_profesional pp ON u.id_usuario = pp.id_usuario
            WHERE u.rol = 'profesional'
            AND u.estado_validacion = 'aprobado'
            ORDER BY u.nombre ASC
        """)
        profesionales = cursor.fetchall()

        return jsonify({
            "profesionales": profesionales,
            "ya_asignado":   ya_asignado,
            "id_solicitado": solicitud['id_profesional'] if solicitud else None
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
# NOTIFICACIONES DEL PROFESIONAL
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

        for n in notificaciones:
            if n['fecha']:
                # FIX: hora Lima en notificaciones
                n['fecha'] = a_hora_lima(n['fecha']).strftime('%d/%m/%Y %H:%M')

        return jsonify({"notificaciones": notificaciones})
    except Exception as e:
        logger.error(f"Error obteniendo notificaciones: {e}")
        return jsonify({"error": "Error del servidor"}), 500
    finally:
        if cursor:   cursor.close()
        if conexion: conexion.close()


# ============================================================
# ATENDER SOLICITUD
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

        cursor.execute("""
            SELECT id_paciente FROM solicitud_apoyo
            WHERE id_solicitud = %s AND id_profesional = %s AND estado = 'pendiente'
        """, (id_solicitud, id_profesional))
        solicitud = cursor.fetchone()
        if not solicitud:
            return jsonify({"success": False, "message": "Solicitud no encontrada."})

        id_paciente = solicitud['id_paciente']
        cursor.execute("""
            INSERT IGNORE INTO asignacion_paciente (id_profesional, id_paciente)
            VALUES (%s, %s)
        """, (id_profesional, id_paciente))
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


# --- Dashboard admin ---
 
@app.route('/admin')
@login_requerido
@solo_admin
def dashboard_admin():
    conexion = None
    cursor   = None
    try:
        conexion = ConexionDb.conexionBaseDeDatos()
        cursor   = conexion.cursor(dictionary=True)
 
        cursor.execute(
            "SELECT nombre, apellidos FROM usuario WHERE id_usuario = %s",
            (session['id_usuario'],)
        )
        datos_admin = cursor.fetchone()
 
        # Postulaciones pendientes con sus documentos
        cursor.execute("""
            SELECT
                u.id_usuario,
                u.nombre,
                u.apellidos,
                u.correo,
                u.edad,
                u.estado_validacion,
                u.fecha_registro,
                pp.cmp,
                pp.especialidad,
                pp.institucion,
                pp.telefono_contacto,
                pp.motivo_rechazo,
                pp.fecha_revision
            FROM usuario u
            INNER JOIN perfil_profesional pp ON u.id_usuario = pp.id_usuario
            WHERE u.rol = 'profesional'
            ORDER BY
                CASE u.estado_validacion
                    WHEN 'pendiente'  THEN 1
                    WHEN 'aprobado'   THEN 2
                    WHEN 'rechazado'  THEN 3
                END,
                u.fecha_registro DESC
        """)
        postulaciones = cursor.fetchall()
 
        # Para cada postulación traer sus documentos
        for p in postulaciones:
            cursor.execute("""
                SELECT tipo, nombre_archivo, url_s3
                FROM documento_postulacion
                WHERE id_usuario = %s
                ORDER BY fecha_subida ASC
            """, (p['id_usuario'],))
            p['documentos'] = cursor.fetchall()
 
            if p.get('fecha_registro'):
                p['fecha_registro'] = a_hora_lima(p['fecha_registro']).strftime('%d/%m/%Y %H:%M')
            if p.get('fecha_revision'):
                p['fecha_revision'] = a_hora_lima(p['fecha_revision']).strftime('%d/%m/%Y %H:%M')
 
        return render_template(
            "dashboard_admin.html",
            admin        = datos_admin,
            postulaciones = postulaciones
        )
 
    except Exception as e:
        logger.error(f"Error cargando dashboard admin: {e}")
        return "Error al cargar el panel de administración."
    finally:
        if cursor:   cursor.close()
        if conexion: conexion.close()
 
 
# --- Aprobar postulación ---
 
@app.route('/api/admin/aprobar', methods=["POST"])
@login_requerido
@solo_admin
def aprobar_postulacion():
    id_usuario     = request.form.get("id_usuario")
    id_admin       = session['id_usuario']
    conexion = None
    cursor   = None
    try:
        conexion = ConexionDb.conexionBaseDeDatos()
        cursor   = conexion.cursor(dictionary=True)
 
        cursor.execute(
            "SELECT nombre, correo FROM usuario WHERE id_usuario = %s AND rol = 'profesional'",
            (id_usuario,)
        )
        usuario = cursor.fetchone()
        if not usuario:
            return jsonify({"success": False, "message": "Usuario no encontrado."})
 
        cursor.execute("""
            UPDATE usuario
            SET estado_validacion = 'aprobado'
            WHERE id_usuario = %s
        """, (id_usuario,))
 
        cursor.execute("""
            UPDATE perfil_profesional
            SET id_admin_revisor = %s,
                fecha_revision   = NOW(),
                motivo_rechazo   = NULL
            WHERE id_usuario = %s
        """, (id_admin, id_usuario))
 
        conexion.commit()
 
        _enviar_email_validacion(
            correo_destino = usuario['correo'],
            nombre         = usuario['nombre'],
            aprobado       = True
        )
 
        logger.info(f"Admin {id_admin} aprobó al profesional {id_usuario}")
        return jsonify({"success": True, "message": f"{usuario['nombre']} fue aprobado."})
 
    except Exception as e:
        logger.error(f"Error aprobando postulación: {e}")
        return jsonify({"success": False, "message": str(e)})
    finally:
        if cursor:   cursor.close()
        if conexion: conexion.close()
 
 
# --- Rechazar postulación ---
 
@app.route('/api/admin/rechazar', methods=["POST"])
@login_requerido
@solo_admin
def rechazar_postulacion():
    id_usuario     = request.form.get("id_usuario")
    motivo         = request.form.get("motivo", "").strip()
    id_admin       = session['id_usuario']
 
    if not motivo:
        return jsonify({"success": False, "message": "Debes indicar un motivo de rechazo."})
 
    conexion = None
    cursor   = None
    try:
        conexion = ConexionDb.conexionBaseDeDatos()
        cursor   = conexion.cursor(dictionary=True)
 
        cursor.execute(
            "SELECT nombre, correo FROM usuario WHERE id_usuario = %s AND rol = 'profesional'",
            (id_usuario,)
        )
        usuario = cursor.fetchone()
        if not usuario:
            return jsonify({"success": False, "message": "Usuario no encontrado."})
 
        cursor.execute("""
            UPDATE usuario
            SET estado_validacion = 'rechazado'
            WHERE id_usuario = %s
        """, (id_usuario,))
 
        cursor.execute("""
            UPDATE perfil_profesional
            SET id_admin_revisor = %s,
                fecha_revision   = NOW(),
                motivo_rechazo   = %s
            WHERE id_usuario = %s
        """, (id_admin, motivo, id_usuario))
 
        conexion.commit()
 
        _enviar_email_validacion(
            correo_destino = usuario['correo'],
            nombre         = usuario['nombre'],
            aprobado       = False,
            motivo         = motivo
        )
 
        logger.info(f"Admin {id_admin} rechazó al profesional {id_usuario} — motivo: {motivo}")
        return jsonify({"success": True, "message": f"Postulación de {usuario['nombre']} rechazada."})
 
    except Exception as e:
        logger.error(f"Error rechazando postulación: {e}")
        return jsonify({"success": False, "message": str(e)})
    finally:
        if cursor:   cursor.close()
        if conexion: conexion.close()
 
 
# --- Ver documentos de un profesional (para el admin) ---
 
@app.route('/api/admin/documentos/<int:id_usuario>')
@login_requerido
@solo_admin
def documentos_profesional(id_usuario):
    conexion = None
    cursor   = None
    try:
        conexion = ConexionDb.conexionBaseDeDatos()
        cursor   = conexion.cursor(dictionary=True)
        cursor.execute("""
            SELECT tipo, nombre_archivo, url_s3,
                   DATE_FORMAT(
                       CONVERT_TZ(fecha_subida, '+00:00', '-05:00'), '%d/%m/%Y %H:%i'
                   ) AS fecha_subida
            FROM documento_postulacion
            WHERE id_usuario = %s
            ORDER BY fecha_subida ASC
        """, (id_usuario,))
        documentos = cursor.fetchall()
        return jsonify({"documentos": documentos})
    except Exception as e:
        logger.error(f"Error obteniendo documentos: {e}")
        return jsonify({"error": "Error del servidor"}), 500
    finally:
        if cursor:   cursor.close()
        if conexion: conexion.close()

# ============================================================
# ARRANQUE
# ============================================================

if __name__ == '__main__':
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(
        debug=debug_mode,
        threaded=True,       # maneja cada request en su propio hilo
        use_reloader=False   # evita WinError 10038 en Windows
    )
