

import streamlit as st
import os
import shutil
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time
import asyncio

# Load environment variables
load_dotenv()

# Streamlit app configuration
st.set_page_config(page_title="Document Q&A", layout="wide")
st.title("Document Q&A")
st.markdown("Upload PDF documents and ask questions about their content. The system will provide answers based solely on the uploaded PDFs.")

# Sidebar for configuration with intelligent defaults
st.sidebar.header("Retrieval Configuration")
st.sidebar.markdown("**Chunk Settings (for better retrieval)**")
chunk_size = st.sidebar.slider(
    "Chunk Size", 
    min_value=500, max_value=2000, value=1200, step=100,
    help="Larger chunks preserve more context but may be less specific"
)
chunk_overlap = st.sidebar.slider(
    "Chunk Overlap", 
    min_value=100, max_value=400, value=300, step=50,
    help="Higher overlap ensures important info isn't split across chunks"
)
retrieval_k = st.sidebar.slider(
    "Retrieved Chunks (k)", 
    min_value=5, max_value=15, value=10, step=1,
    help="More chunks = better recall but potentially more noise"
)

# Advanced settings
with st.sidebar.expander("Advanced Settings"):
    similarity_threshold = st.slider(
        "Similarity Score Threshold", 
        min_value=0.0, max_value=1.0, value=0.1, step=0.05,
        help="Lower = more permissive retrieval"
    )
    enable_mmr = st.checkbox(
        "Enable MMR (Maximal Marginal Relevance)", 
        value=True,
        help="Reduces redundancy in retrieved chunks"
    )

# File uploader for PDFs
st.sidebar.header("Upload PDF Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload one or more PDF files"
)

# Directory for uploaded files
UPLOAD_DIR = "./uploaded_pdfs"

# Create upload directory if it doesn't exist
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Process uploaded files
if uploaded_files:
    # Clear the upload directory to avoid mixing old and new files
    for file in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, file)
        if os.path.isfile(file_path):
            os.unlink(file_path)
    
    for uploaded_file in uploaded_files:
        # Check file size (50 MB limit = 50 * 1024 * 1024 bytes)
        if uploaded_file.size > 50 * 1024 * 1024:
            st.sidebar.error(f"File {uploaded_file.name} exceeds 50 MB limit")
            continue
        
        # Save the uploaded file to the temporary directory
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success(f"Uploaded {uploaded_file.name}")

# Validate API keys
def validate_api_keys():
    """Validate that required API keys are present"""
    groq_key = os.getenv('GROQ_API_KEY')
    google_key = os.getenv("GOOGLE_API_KEY")
    
    if not groq_key:
        st.error("❌ GROQ_API_KEY not found in environment variables")
        return None, None
    if not google_key:
        st.error("❌ GOOGLE_API_KEY not found in environment variables") 
        return groq_key, None
    
    # Set Google API key for the embeddings
    os.environ["GOOGLE_API_KEY"] = google_key
    return groq_key, google_key

# Cache the LLM initialization
@st.cache_resource
def get_llm(groq_api_key):
    """Initialize and cache the Groq LLM"""
    return ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Cache the embeddings initialization  
@st.cache_resource
def get_embeddings():
    """Initialize and cache the Google embeddings model"""
    try:
        # Try to get existing event loop
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # Create new event loop if none exists
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Strict grounding prompt template with better instructions
@st.cache_resource
def get_prompt_template():
    """Create and cache the strict grounding prompt template"""
    return ChatPromptTemplate.from_template(
        """
        You are a precise research assistant that answers questions based ONLY on the provided context.
        
        CRITICAL INSTRUCTIONS:
        1. Use ONLY the information explicitly stated in the context below
        2. If the answer is not found in the context, respond EXACTLY: "Answer not found in the provided documents."
        3. When you find relevant information, quote directly from the context
        4. Be comprehensive - if multiple relevant pieces are in the context, include them all
        5. Do not use external knowledge, assumptions, or inferences beyond what's stated
        
        <context>
        {context}
        </context>
        
        Question: {input}
        
        Based strictly on the context above, provide your answer:
        """
    )

# Enhanced vector store creation with better document processing
@st.cache_resource
def create_vector_store(_embeddings, chunk_size, chunk_overlap):
    """Load documents, split them intelligently, and create FAISS vector store"""
    
    # Check if directory exists
    if not os.path.exists(UPLOAD_DIR):
        st.error(f"❌ Directory '{UPLOAD_DIR}' not found. Please upload PDF files.")
        return None, None
    
    # Load all documents from directory
    with st.spinner("📚 Loading PDF documents..."):
        loader = PyPDFDirectoryLoader(UPLOAD_DIR)
        docs = loader.load()
    
    if not docs:
        st.error(f"❌ No PDF documents found in '{UPLOAD_DIR}' directory.")
        return None, None
    
    st.info(f"✅ Loaded {len(docs)} document pages")
    
    # Enhanced text splitting for better retrieval
    with st.spinner("✂️ Splitting documents with optimized chunking..."):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # Better separators for academic/historical texts
            separators=[
                "\n\n",           # Paragraph breaks
                "\nCHAPTER",      # Chapter headings  
                "\nSection",      # Section headings
                "\n\n",           # Double newlines
                "\n",             # Single newlines
                ". ",             # Sentence endings
                ", ",             # Clause breaks
                " ",              # Word boundaries
                ""                # Character level (last resort)
            ],
            length_function=len,
            is_separator_regex=False,
        )
        
        # Process ALL documents (removed [:20] limit)
        final_documents = text_splitter.split_documents(docs)
        
        # Add enhanced metadata for better tracking
        for i, doc in enumerate(final_documents):
            doc.metadata['chunk_id'] = i
            doc.metadata['chunk_size'] = len(doc.page_content)
            # Extract potential headings from chunk start
            first_lines = doc.page_content[:200].split('\n')[:3]
            doc.metadata['content_preview'] = ' | '.join([line.strip() for line in first_lines if line.strip()])
    
    st.info(f"✅ Created {len(final_documents)} optimized text chunks")
    
    # Create vector embeddings with progress tracking
    with st.spinner("🧠 Creating vector embeddings (this may take a while)..."):
        # Process in batches for large document sets
        if len(final_documents) > 1000:
            st.info("Large document set detected - processing in batches...")
            batch_size = 100
            all_texts = []
            all_metadatas = []
            
            for i in range(0, len(final_documents), batch_size):
                batch = final_documents[i:i+batch_size]
                batch_texts = [doc.page_content for doc in batch]
                batch_metadatas = [doc.metadata for doc in batch]
                all_texts.extend(batch_texts)
                all_metadatas.extend(batch_metadatas)
                
   
            vectors = FAISS.from_texts(all_texts, _embeddings, metadatas=all_metadatas)
        else:
            vectors = FAISS.from_documents(final_documents, _embeddings)
    
    st.success(f"✅ Vector store created with {len(final_documents)} chunks")
    return vectors, final_documents

# Enhanced answer extraction
def extract_answer(response):
    """Robustly extract answer from different chain output formats"""
    possible_keys = ['answer', 'result', 'output_text', 'response']
    
    if isinstance(response, dict):
        for key in possible_keys:
            if key in response:
                return response[key]
        return str(response)
    
    return str(response)

# Main application logic
def main():
    # Validate API keys first
    groq_key, google_key = validate_api_keys()
    if not groq_key or not google_key:
        st.stop()
    
    # Initialize cached resources
    llm = get_llm(groq_key)
    embeddings = get_embeddings()
    prompt_template = get_prompt_template()
    
    # Document Processing Section
    st.header("Document Processing")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("Create Document Embeddings", type="primary"):
            try:
                vectors, documents = create_vector_store(embeddings, chunk_size, chunk_overlap)
                if vectors and documents:
                    st.session_state.vectors = vectors
                    st.session_state.documents = documents
                    st.session_state.embedding_complete = True
            except Exception as e:
                st.error(f"❌ Error creating embeddings: {str(e)}")
    
    with col2:
        if hasattr(st.session_state, 'embedding_complete') and st.session_state.embedding_complete:
            st.success(f"✅ Ready: {len(st.session_state.documents)} chunks")
    
    # Q&A Section
    st.header("Question & Answer")
    
    user_question = st.text_input(
        "Ask a question about your documents:", 
        placeholder="According to the provided text, what qualities did the Romans need to achieve and maintain mastery?",
        help="The AI will answer based only on the content in your PDFs"
    )
    
    if user_question and user_question.strip():
        if not hasattr(st.session_state, 'embedding_complete') or not st.session_state.embedding_complete:
            st.warning("⚠️ Please create document embeddings first.")
        else:
            try:
                # Configure retriever with advanced options
                search_kwargs = {"k": retrieval_k}
                
                if enable_mmr:
                    # Use MMR for diversity in retrieved chunks
                    retriever = st.session_state.vectors.as_retriever(
                        search_type="mmr",
                        search_kwargs={
                            "k": retrieval_k,
                            "fetch_k": retrieval_k * 2,  # Fetch more candidates for MMR
                            "lambda_mult": 0.7  # Balance between relevance and diversity
                        }
                    )
                else:
                    retriever = st.session_state.vectors.as_retriever(search_kwargs=search_kwargs)
                
                # Create and execute retrieval chain
                document_chain = create_stuff_documents_chain(llm, prompt_template)
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                
                # Time the operation
                with st.spinner("🤔 Analyzing documents and generating answer..."):
                    start_time = time.perf_counter()
                    response = retrieval_chain.invoke({'input': user_question})
                    end_time = time.perf_counter()
                
                # Display results
                response_time = end_time - start_time
                st.info(f"⏱️ Response time: {response_time:.2f} seconds")
                
                # Show answer
                answer = extract_answer(response)
                st.success("**Answer:**")
                st.write(answer)
                
            except Exception as e:
                st.error(f"❌ Error processing question: {str(e)}")
                st.exception(e)  # Show full traceback for debugging

# Additional tips section
with st.sidebar.expander("💡 Retrieval Tips"):
    st.markdown("""
    **For better results:**
    - Use specific terms from your documents
    - Ask about explicit content, not inferences
    - Try different chunk sizes for different document types
    - Use MMR for diverse information gathering
    
    **Optimal settings:**
    - Academic texts: 1200-1500 chunk size
    - Technical docs: 800-1200 chunk size  
    - Narrative texts: 1000-1500 chunk size
    - High overlap (300+) preserves context
    """)

if __name__ == "__main__":
    main()