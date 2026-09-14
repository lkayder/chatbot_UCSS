import os
import streamlit as st
from dotenv import load_dotenv

# Importaciones de LangChain
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Cargar variables de entorno (API Key)
load_dotenv()

# Configuración de la página
st.set_page_config(page_title="Asistente Virtual UCSS", page_icon="🎓", layout="centered")
st.title("🎓 Asistente Virtual UCSS - Transparencia")

# Asegurarnos de que la API Key esté presente
if not os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") == "pega_tu_llave_aqui_sin_comillas":
    st.error("⚠️ Falla de configuración: Por favor, asegúrate de haber pegado tu API KEY de Google en el archivo .env")
    st.stop()

# --- FUNCIONES DE IA Y BASE DE DATOS ---
@st.cache_resource(show_spinner="Preparando el cerebro del bot...")
def inicializar_bot():
    url = "https://www.ucss.edu.pe/nosotros/transparencia"
    db_directory = "chroma_db_local"
    
    # Modelo de embeddings LOCAL (gratuito, súper rápido y no depende de la versión de la API de Google)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Si la base de datos ya existe, simplemente la cargamos
    if os.path.exists(db_directory):
        vectorstore = Chroma(persist_directory=db_directory, embedding_function=embeddings)
    else:
        # Si no existe, descargamos la página y creamos la base
        loader = WebBaseLoader(url)
        # Desactivamos SSL verify para evitar errores de certificado
        loader.requests_kwargs = {'verify': False}
        documentos = loader.load()
        
        # Dividimos el texto en pedazos más pequeños
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documentos)
        
        # Guardamos en la base de datos vectorial
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=db_directory)

    # Configuramos el modelo de lenguaje de Gemini para RESPONDER
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.3)
    
    # Creamos el prompt (instrucciones para el bot)
    system_prompt = (
        "Eres el asistente virtual inteligente de la Universidad Católica Sedes Sapientiae (UCSS). "
        "Tu objetivo principal es ayudar a los estudiantes y usuarios respondiendo sus dudas.\n\n"
        "REGLAS PARA RESPONDER:\n"
        "1. Primero, revisa la 'Información recuperada' abajo. Si la respuesta está ahí, úsala.\n"
        "2. Si la información recuperada no contiene la respuesta exacta (o está vacía), USA TU CONOCIMIENTO GENERAL sobre la UCSS, sus autoridades, facultades y funcionamiento para dar una respuesta útil y real.\n"
        "3. Si te preguntan de dónde sacas la información, diles que tu fuente principal es el portal web: https://www.ucss.edu.pe/nosotros/transparencia\n"
        "4. Sé siempre muy amable, claro y profesional.\n\n"
        "Información recuperada del portal web:\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Creamos la cadena (chain) de respuesta
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

# Inicializamos el bot (esta función se guarda en caché y solo corre una vez)
try:
    bot_chain = inicializar_bot()
except Exception as e:
    st.error(f"Ocurrió un error al cargar la información: {e}")
    st.stop()

# --- INTERFAZ DE CHAT ---

st.markdown("¡Hola! Soy el asistente virtual de la Universidad Católica Sedes Sapientiae. Puedo responder preguntas basadas en la información pública del portal de transparencia de la universidad.")

# Inicializar historial
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []
    st.session_state.mensajes.append({
        "role": "assistant",
        "content": "¿En qué te puedo ayudar hoy? Pregúntame sobre la universidad."
    })

# Mostrar los mensajes del historial
for mensaje in st.session_state.mensajes:
    with st.chat_message(mensaje["role"]):
        st.markdown(mensaje["content"])

# Capturar input del usuario
prompt = st.chat_input("Escribe tu pregunta aquí...")

if prompt:
    # 1. Mostrar mensaje del usuario
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.mensajes.append({"role": "user", "content": prompt})
    
    # 2. Generar y mostrar respuesta de la IA
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            try:
                # Le preguntamos a la IA usando nuestra cadena RAG
                respuesta = bot_chain.invoke({"input": prompt})
                texto_respuesta = respuesta["answer"]
                st.markdown(texto_respuesta)
                
                # Guardamos la respuesta en el historial
                st.session_state.mensajes.append({"role": "assistant", "content": texto_respuesta})
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    error_msg = "Ups, estás haciendo preguntas muy rápido y hemos alcanzado el límite gratuito de Google (protección anti-spam). Por favor, espera unos segundos y vuelve a intentarlo. ⏳"
                else:
                    error_msg = f"Uy, tuve un problema al procesar tu pregunta: {error_str}"
                
                st.warning(error_msg)
                # No guardamos el error en el historial para que no ensucie la conversación
