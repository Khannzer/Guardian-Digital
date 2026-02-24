from flask import Flask, render_template, jsonify, request, redirect, url_for
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from src.prompt import system_prompt # Importando tu prompt
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
# LOGIN
# ----------------------------

@app.route("/")
def home():
    return render_template("inicio.html")  # tu login


@app.route("/login", methods=["POST"])
def login():
    username = request.form["username"]
    password = request.form["password"]

    if username == "admin" and password == "1234":
        return redirect(url_for("chat_page"))
    else:
        return "Usuario o contraseña incorrectos"


# ----------------------------
# CHAT PAGE
# ----------------------------

@app.route("/chat")
def chat_page():
    return render_template("chat.html")

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
# REGISTER
# ----------------------------
@app.route("/registro")
def register_page():
    return render_template("registro.html")

# ----------------------------
@app.route("/inicio")
def inicio_page():
    return render_template("inicio.html")
# ----------------------------

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
    
    
