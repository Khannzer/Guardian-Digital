from flask import Flask, render_template, jsonify, request, redirect, url_for
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
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

chatModel = ChatOpenAI(model="gpt-4o")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

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
    msg = request.form["msg"]
    response = rag_chain.invoke({"input": msg})
    return str(response["answer"])


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
    
    
