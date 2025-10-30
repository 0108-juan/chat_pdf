import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import platform

# Configuración de página
st.set_page_config(
    page_title="RAG - Chat con PDF",
    page_icon="📚",
    layout="wide"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .upload-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 3px dashed #3B82F6;
        text-align: center;
        margin: 1rem 0;
    }
    .api-section {
        background: linear-gradient(135deg, #E0F7FA 0%, #B2EBF2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #00ACC1;
        margin: 1rem 0;
    }
    .response-box {
        background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #10B981;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #E2E8F0;
        margin: 0.5rem 0;
        text-align: center;
    }
    .sidebar-info {
        background-color: #F1F5F9;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1E3A8A;
        margin: 1rem 0;
    }
    .tech-badge {
        background-color: #1E3A8A;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.7rem;
        display: inline-block;
        margin: 0.1rem;
    }
    .question-input {
        background-color: #F8FAFC;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

# Header principal
st.markdown('<h1 class="main-title">📚 Chat Inteligente con PDFs</h1>', unsafe_allow_html=True)
st.markdown("### 💬 Generación Aumentada por Recuperación (RAG)")

# Información del sistema
col_sys1, col_sys2, col_sys3 = st.columns(3)
with col_sys1:
    st.metric("Python", platform.python_version())
with col_sys2:
    st.metric("Framework", "LangChain")
with col_sys3:
    st.metric("Modelo", "GPT-4")

# Logo de la aplicación
col_img, col_desc = st.columns([1, 2])
with col_img:
    try:
        image = Image.open('Chat_pdf.png')
        st.image(image, use_container_width=True, caption="Asistente IA para PDFs")
    except Exception as e:
        st.info("📚 Icono de la aplicación")

with col_desc:
    st.markdown("""
    ### 🚀 Acerca de esta App
    Convierte tus documentos PDF en conversaciones inteligentes usando **RAG** 
    (Retrieval-Augmented Generation) con tecnología de OpenAI.
    
    **Tecnologías:**
    <span class="tech-badge">OpenAI GPT-4</span>
    <span class="tech-badge">LangChain</span>
    <span class="tech-badge">FAISS</span>
    <span class="tech-badge">Streamlit</span>
    """, unsafe_allow_html=True)

# Sidebar mejorado
with st.sidebar:
    st.markdown("### 🔍 Información del Agente")
    st.markdown("""
    <div class="sidebar-info">
    <p>Este agente de IA te ayudará a:</p>
    <ul>
    <li>📖 Analizar documentos PDF</li>
    <li>🔍 Buscar información específica</li>
    <li>💡 Resumir contenido</li>
    <li>❓ Responder preguntas</li>
    </ul>
    <p><em>Powered by OpenAI & LangChain</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 💡 Cómo usar")
    st.markdown("""
    <div class="sidebar-info">
    1. 🔑 Ingresa tu API Key de OpenAI
    2. 📄 Sube tu archivo PDF
    3. ⏳ Espera el procesamiento
    4. ❓ Haz tus preguntas
    5. 💬 Recibe respuestas inteligentes
    </div>
    """, unsafe_allow_html=True)

# Sección de API Key
st.markdown("### 🔑 Configuración de OpenAI")
st.markdown('<div class="api-section">', unsafe_allow_html=True)
ke = st.text_input(
    'Ingresa tu Clave de OpenAI API', 
    type="password",
    placeholder="sk-...",
    help="Obtén tu API key en https://platform.openai.com/api-keys"
)
if ke:
    os.environ['OPENAI_API_KEY'] = ke
    st.success("✅ API Key configurada correctamente")
else:
    st.warning("⚠️ Ingresa tu clave de API para continuar")
st.markdown('</div>', unsafe_allow_html=True)

# Sección de carga de PDF
st.markdown("### 📄 Carga tu Documento PDF")
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
pdf = st.file_uploader(
    "Arrastra o selecciona tu archivo PDF", 
    type="pdf",
    help="Sube un documento PDF para analizar con IA"
)
st.markdown('</div>', unsafe_allow_html=True)

# Procesamiento del PDF
if pdf is not None and ke:
    try:
        with st.spinner("📖 Extrayendo texto del PDF..."):
            # Extraer texto del PDF
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            # Mostrar métricas de extracción
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                <h3>📄 Páginas</h3>
                <h2 style="color: #1E3A8A;">{len(pdf_reader.pages)}</h2>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                <h3>🔤 Caracteres</h3>
                <h2 style="color: #1E3A8A;">{len(text)}</h2>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                <h3>📊 Estado</h3>
                <h4 style="color: #10B981;">Extraído</h4>
                </div>
                """, unsafe_allow_html=True)
        
        with st.spinner("🔪 Dividiendo documento en fragmentos..."):
            # Dividir texto en chunks
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=500,
                chunk_overlap=20,
                length_function=len
            )
            chunks = text_splitter.split_text(text)
            
            st.success(f"✅ Documento dividido en {len(chunks)} fragmentos")
        
        with st.spinner("🧠 Creando base de conocimiento con embeddings..."):
            # Crear embeddings y base de conocimiento
            embeddings = OpenAIEmbeddings()
            knowledge_base = FAISS.from_texts(chunks, embeddings)
            st.success("✅ Base de conocimiento creada exitosamente")
        
        # Interfaz de preguntas
        st.markdown("### ❓ Haz tu Pregunta")
        st.markdown('<div class="question-input">', unsafe_allow_html=True)
        user_question = st.text_area(
            "Escribe qué quieres saber sobre el documento:",
            placeholder="Ej: ¿Cuál es el resumen del documento? ¿Qué se dice sobre [tema específico]? ¿Cuáles son los puntos principales?",
            height=100,
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Procesar pregunta cuando se envía
        if user_question:
            with st.spinner("🔍 Buscando información relevante..."):
                docs = knowledge_base.similarity_search(user_question)
                
                st.info(f"📚 Encontrados {len(docs)} fragmentos relevantes")
            
            with st.spinner("💭 Generando respuesta con IA..."):
                # Usar modelo actual
                llm = OpenAI(temperature=0, model_name="gpt-4o")
                
                # Cargar cadena de QA
                chain = load_qa_chain(llm, chain_type="stuff")
                
                # Ejecutar la cadena
                response = chain.run(input_documents=docs, question=user_question)
                
                # Mostrar la respuesta
                st.markdown("### 💬 Respuesta del Asistente")
                st.markdown(f'<div class="response-box">{response}</div>', unsafe_allow_html=True)
                    
    except Exception as e:
        st.error(f"❌ Error al procesar el PDF: {str(e)}")
elif pdf is not None and not ke:
    st.warning("🔑 Por favor ingresa tu clave de API de OpenAI para continuar")
else:
    st.info("📄 Por favor carga un archivo PDF para comenzar el análisis")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6B7280;'>
<p><strong>🤖 Chat con PDFs - RAG System</strong> • Desarrollado con LangChain y OpenAI</p>
<p>Transforma tus documentos en conversaciones inteligentes</p>
</div>
""", unsafe_allow_html=True)
