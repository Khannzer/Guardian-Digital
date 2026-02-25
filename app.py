from flask import Flask, render_template, jsonify, request, redirect, url_for
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from src.prompt import system_prompt # Importando tu prompt
from conexionDb.conexionDb import ConexionDb
from werkzeug.security import check_password_hash
import os


app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = download_embeddings()
index_name = "guardian-digital"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

# ---------------------------------------------------------
# 1. DEFINICIÓN DE LA ESTRUCTURA DE RESPUESTA (El "Function Calling")
# Le decimos a la IA exactamente qué campos debe llenar
# ---------------------------------------------------------
class RespuestaGuardian(BaseModel):
    answer: str = Field(description="La respuesta empática y de apoyo para el usuario, basada en la guía mhGAP.")
    riesgo_inminente: bool = Field(description="True (Verdadero) si el usuario presenta riesgo suicida inminente, intención explícita de hacerse daño o desesperanza extrema. False (Falso) en caso contrario.")

# ---------------------------------------------------------
# 2. CONFIGURACIÓN DEL MODELO CON SALIDA ESTRUCTURADA
# Forzamos al modelo gpt-4o a devolver siempre un objeto con el formato de la clase RespuestaGuardian
# ---------------------------------------------------------
# Nota: Bajamos un poco la temperatura para que la detección de riesgo sea más precisa y menos alucinada
chatModel = ChatOpenAI(model="gpt-4o", temperature=0.2)
structured_llm = chatModel.with_structured_output(RespuestaGuardian)

# 3. CONFIGURACIÓN DEL PROMPT
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# 4. FUNCIÓN AUXILIAR PARA FORMATEAR LOS DOCUMENTOS (Contexto de Pinecone)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ---------------------------------------------------------
# 5. NUEVA CADENA RAG USANDO LCEL (LangChain Expression Language)
# Es la forma moderna recomendada por LangChain
# ---------------------------------------------------------
rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | structured_llm
)




# ----------------------------
# TE MANDA AUTOMATICAMENTE AL INICIO
# ----------------------------
@app.route("/")
def home():
    return render_template("inicio.html")  # tu login

# ----------------------------
# CHAT PAGE
# ----------------------------
@app.route("/chat")
def chat_page():
    return render_template("chat.html")

# ----------------------------
# LOGIN
# ----------------------------
@app.route("/login", methods=["POST"])
def login():
    nombre = request.form.get("txtNombre")
    contrasenia = request.form.get("txtContrasenia")

    try:
        conexion = ConexionDb.conexionBaseDeDatos()
        cursor = conexion.cursor(dictionary=True)

        sql = "SELECT * FROM usuario WHERE nombre = %s AND contrasenia = %s"
        cursor.execute(sql, (nombre, contrasenia))
        usuario = cursor.fetchone()

        if usuario:
            return redirect(url_for("chat_page"))
        else:
            return "Usuario o contraseña incorrectos"

    except Exception as e:
        return f"Usuario o contraseña incorrectos. Detalle del error: {e}"
# ---------------------------

@app.route("/get", methods=["POST"])
def get_response():
    try:
        msg = request.form["msg"]
        
        # Intentamos obtener la respuesta de la IA
        response_obj = rag_chain.invoke(msg)
        
        return jsonify({
            "answer": response_obj.answer,
            "riesgo_inminente": response_obj.riesgo_inminente
        })
        
    except Exception as e:
        # SI ALGO FALLA (ej. OpenAI se cae, no hay internet en el servidor)
        # Imprimimos el error en consola para ti (el desarrollador)
        print(f"Error en el backend: {e}")
        
        # Le enviamos al usuario una respuesta segura de respaldo y activamos la tarjeta
        return jsonify({
            "answer": "Lo siento, estoy experimentando problemas técnicos en este momento. Por favor, si sientes que estás en una crisis o necesitas ayuda urgente, no esperes y contacta a los servicios de emergencia de tu localidad.",
            "riesgo_inminente": True # Activamos la tarjeta por precaución
        })

# ----------------------------
# REGISTRA LOS DEPARTAMENTOS EN EL FORMULARIO DE REGITRO DE USUARIOS
# ----------------------------
@app.route("/registroUsuario")
def registro():
    conexion = ConexionDb.conexionBaseDeDatos()
    cursor = conexion.cursor(dictionary=True)

    cursor.execute("SELECT * FROM departamento")
    departamentos = cursor.fetchall()
    
    cursor.close()
    conexion.close()

    return render_template("registroUsuario.html", departamentos=departamentos)

# ----------------------------
# SE USA PARA DIRIGIR AL INICIO DE UNA PESTAÑA A OTRA
# ----------------------------
@app.route("/inicio")
def inicio_page():
    return render_template("inicio.html")
# ----------------------------


# ----------------------------
# CARGA DE LOS DEPARTAMENTOS
# ----------------------------
@app.route("/provincias/<int:id_departamento>")
def obtener_provincias(id_departamento):
    conexion = ConexionDb.conexionBaseDeDatos()
    cursor = conexion.cursor(dictionary=True)

    sql = "SELECT id_provincia, nombre FROM provincia WHERE id_departamento = %s"
    cursor.execute(sql, (id_departamento,))
    provincias = cursor.fetchall()
    
    cursor.close()
    conexion.close()

    return jsonify(provincias)

# ----------------------------
# CARGA DE LOS DISTRITOS
# ----------------------------
@app.route("/distritos/<int:id_provincia>")
def obtener_distritos(id_provincia):
    conexion = ConexionDb.conexionBaseDeDatos()
    cursor = conexion.cursor(dictionary=True)

    sql = "SELECT id_distrito, nombre FROM distrito WHERE id_provincia = %s"
    cursor.execute(sql, (id_provincia,))
    distritos = cursor.fetchall()
    
    cursor.close()
    conexion.close()

    return jsonify(distritos)


# ----------------------------
# REGISTRAR AL USUARIO
# ----------------------------
@app.route("/registrar", methods=["POST"])
def registrar_usuario():
    try:
        conexion = ConexionDb.conexionBaseDeDatos()
        cursor = conexion.cursor()

        nombre = request.form["txtNombre"]
        apellidos = request.form["txtApellidos"]
        contrasenia = request.form["txtContrasenia"]
        correo = request.form["txtCorreo"]
        edad = request.form["txtEdad"]
        #departamento = request.form["selectDepartamento"]
        #provincia = request.form["selectProvincia"]
        distrito = request.form["selectDistrito"]

        sql = """
            INSERT INTO usuario
            (nombre, apellidos, correo, contrasenia, edad, id_distrito)
            VALUES (%s, %s, %s, %s, %s, %s)
        """

        valores = (nombre, apellidos, correo, contrasenia, edad, distrito)

        cursor.execute(sql, valores)
        conexion.commit()

        cursor.close()
        conexion.close()

        return jsonify({"success": True, "message": "Usuario registrado correctamente"})

    except Exception as e:
        print("Error al registrar:", e)
        return jsonify({"success": False, "message": "No se pudo registrar el usuario"})






if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
