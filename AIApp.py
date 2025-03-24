import os
import io
import fitz  # PyMuPDF for reading PDFs
import faiss  # Vector database
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from reportlab.pdfgen import canvas
from gtts import gTTS  # New import for text-to-speech

# Load environment variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

# Load embedding model
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def create_faiss_index(chunks):
    vectors = np.array([embedder.encode(chunk) for chunk in chunks]).astype('float32')
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index, chunks

def retrieve_relevant_chunks(query, index, chunks, top_k=3):
    query_vector = np.array([embedder.encode(query)]).astype('float32')
    distances, indices = index.search(query_vector, top_k)
    return [chunks[i] for i in indices[0]]

deepseek_engine = ChatGroq(api_key=groq_api_key, model="deepseek-r1-distill-llama-70b", temperature=0.3)

def generate_answer(context, query):
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    response = deepseek_engine.invoke(prompt)
    return response

# Helper function to create PDF from answer text
def save_answer_as_pdf(answer):
    if not isinstance(answer, str):
        if hasattr(answer, 'content'):
            answer_text = answer.content
        else:
            answer_text = str(answer)
    else:
        answer_text = answer

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer)
    pdf.setFont("Helvetica", 12)
    text_object = pdf.beginText(40, 800)
    for line in answer_text.splitlines():
        text_object.textLine(line)
    pdf.drawText(text_object)
    pdf.showPage()
    pdf.save()
    buffer.seek(0)
    return buffer

# Helper function to get plain text from answer
def get_answer_as_text(answer):
    if not isinstance(answer, str):
        if hasattr(answer, 'content'):
            return answer.content
        else:
            return str(answer)
    else:
        return answer

# New Feature: Helper function to convert answer text to audio using gTTS
def get_answer_audio(answer, lang="en"):
    # Convert answer to string if not already
    if not isinstance(answer, str):
        if hasattr(answer, 'content'):
            answer_text = answer.content
        else:
            answer_text = str(answer)
    else:
        answer_text = answer
    
    # Create audio using gTTS
    tts = gTTS(text=answer_text, lang=lang)
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer

# Custom CSS styling
st.markdown("""
<style>
    /* Existing styles */
    .main {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #2d2d2d;
    }
    .stTextInput textarea {
        color: #ffffff !important;
    }
    .stSelectbox div[data-baseweb="select"] {
        color: white !important;
        background-color: #3d3d3d !important;
    }
    .stSelectbox svg {
        fill: white !important;
    }
    .stSelectbox option {
        background-color: #2d2d2d !important;
        color: white !important;
    }
    div[role="listbox"] div {
        background-color: #2d2d2d !important;
        color: white !important;
    }
    .chat-message {
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .chat-message.user {
        background-color: #4a4a4a;
    }
    .chat-message.bot {
        background-color: #3d3d3d;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 AI-Powered PDF Insights Analyst AI Agent APP")
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
st.caption("🚀 Your AI Assistant To Solve Pdf Q/N Superpowers🚨")

# Sidebar configuration
with st.sidebar:
    st.header("⚙ Configuration")
    selected_model = st.selectbox(
        "Choose Model",
        ["deepseek-r1-distill-llama-70b"],
        index=0
    )
    st.divider()
    st.markdown("### Model Capabilities")
    st.markdown("""
    - 🚨AI Expert
    - 🐞 Q/N Assistant
    - 📝 Detail Documentation
    - 💡 Solution Design
    """)
    st.divider()
    st.markdown("Built with [Groq](https://groq.com/) | [LangChain](https://python.langchain.com/)")
    st.markdown("👨👨‍💻Developer:- Abhishek❤️Kumar")
    
    developer_path = "pic.jpg"  # Ensure this file is in the same directory as your script
try:
    st.sidebar.image(developer_path)
except FileNotFoundError:
    st.sidebar.warning("pic.jpg file not found. Please check the file path.")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success("PDF uploaded successfully!😊")
    text = extract_text_from_pdf("temp.pdf")
    chunks = chunk_text(text)
    index, chunk_list = create_faiss_index(chunks)
    st.session_state["index"] = index
    st.session_state["chunks"] = chunk_list
    st.session_state["ready"] = True

if "ready" in st.session_state:
    query = st.text_input("👨‍💻Ask a question about the PDF❤:")
    if query:
        relevant_chunks = retrieve_relevant_chunks(query, st.session_state["index"], st.session_state["chunks"])
        context = "\n".join(relevant_chunks)
        answer = generate_answer(context, query)
        st.write("### Answer:")
        st.write(answer)
        
        # Download Answer as PDF button (existing functionality)
        pdf_buffer = save_answer_as_pdf(answer)
        st.download_button("Download Answer as PDF", pdf_buffer, file_name="answer.pdf", mime="application/pdf")
        
        # Download Answer as Text File button (existing functionality)
        answer_text = get_answer_as_text(answer)
        st.download_button("Download Answer as Text", answer_text, file_name="answer.txt", mime="text/plain")
        
        # New Feature: Play Answer Audio button
        if st.button("Play Answer Audio"):
            audio_buffer = get_answer_audio(answer)
            st.audio(audio_buffer, format="audio/mp3")
        
        # Toggle to show the context used for generating the answer
        if st.checkbox("Show Context Used"):
            st.markdown("#### Context:")
            st.text(context)
