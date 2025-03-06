import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
import re

# Load environment variables
load_dotenv()

# Set Streamlit page config
st.set_page_config(page_title="Chat with your PDF", page_icon="📚", layout="wide")

# Sidebar styling
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>🔍 Model Selection</h2>", unsafe_allow_html=True)
    
    models = {
        "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"},
        "llama3-8b-8192": {"name": "LLaMA3-8b-8192", "tokens": 8192, "developer": "Meta"},
        "gemma2-9b-it": {"name": "gemma2-9b-it", "tokens": 8192, "developer": "Google"},
        "deepseek-r1-distill-llama-70b": {"name": "deepseek-r1-distill-llama-70b", "tokens":16384, "developer": "Deepseek"},
    }
    
    model_option = st.selectbox("Choose a model:", options=list(models.keys()), format_func=lambda x: models[x]["name"], index=0)
    max_tokens = st.slider("Max Tokens:", min_value=512, max_value=models[model_option]["tokens"], value=1024, step=512)
    
    st.markdown("<h2 style='text-align: center;'>📂 Upload PDF</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], help="Select a PDF to chat with.")

# Main header with gradient effect
st.markdown(
    """
    <h1 style="text-align: center; background: linear-gradient(90deg, #ff7e5f, #feb47b); -webkit-background-clip: text; color: transparent;">📚 Talk to Your PDF</h1>
    """,
    unsafe_allow_html=True
)

# Initialize Groq client
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("API key not found! Please configure it.")
    st.stop()
client = Groq(api_key=api_key)

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Process uploaded PDF
if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        loader = PDFPlumberLoader("temp.pdf")
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        documents = text_splitter.split_documents(docs)
        
        embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(documents, embedder)
        vectorstore.save_local("faiss_index")
        
        st.session_state.vectorstore = vectorstore
        st.success("✅ PDF processed successfully! You can start chatting.")

# Load FAISS index if it exists
if st.session_state.vectorstore is None and os.path.exists("faiss_index"):
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    st.session_state.vectorstore = FAISS.load_local("faiss_index", embedder, allow_dangerous_deserialization=True)

retriever = st.session_state.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2}) if st.session_state.vectorstore else None

# Chat history UI
st.markdown("<h3 style='text-align: center;'>💬 Chat History</h3>", unsafe_allow_html=True)
chat_container = st.container()

with chat_container:
    for msg in st.session_state.get("messages", []):
        role, content = msg["role"], msg["content"]
        with st.chat_message(role, avatar='💡' if role == "assistant" else '👨‍💻'):
            st.markdown(content)

# Function to clean response

def clean_response(response):
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

# Chat input
if retriever and (prompt := st.chat_input("Ask a question about your PDF...")):
    st.session_state.setdefault("messages", []).append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar='👨‍💻'):
        st.markdown(prompt)
    
    # Retrieve relevant PDF context
    retrieved_docs = retriever.get_relevant_documents(prompt)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    formatted_prompt = f"Use only the following context to answer the question: {context}\nQuestion: {prompt}\nAnswer:"
    
    # Fetch response from Groq API
    try:
        response = client.chat.completions.create(
            model=model_option,
            messages=[{"role": "user", "content": formatted_prompt}],
            max_tokens=max_tokens
        )
        bot_response = clean_response(response.choices[0].message.content)
    except Exception as e:
        bot_response = f"Error: {e}"
    
    # Display response
    st.session_state["messages"].append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant", avatar="💡"):
        st.markdown(bot_response)
