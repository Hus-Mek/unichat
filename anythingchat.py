import streamlit as st
import pandas as pd
import io
import asyncio
import time
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq

# RAG-Anything imports
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc
from lightrag import LightRAG

# ============================================
# SESSION STATE
# ============================================
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "last_request_time" not in st.session_state:
    st.session_state.last_request_time = 0
if "rag_instance" not in st.session_state:
    st.session_state.rag_instance = None

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(page_title="Uni Assistant - RAG-Anything", layout="wide")
st.title("University Chatbot - Powered by RAG-Anything 🚀")

# ============================================
# API KEY SETUP
# ============================================
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    st.error("⚠️ API key not configured. Please contact the administrator.")
    st.stop()

groq_client = Groq(api_key=GROQ_API_KEY)

# ============================================
# EMBEDDING MODEL (Cached)
# ============================================
@st.cache_resource
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = get_embedder()

# ============================================
# LLM FUNCTION (Groq)
# ============================================
def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    """Groq LLM for text processing"""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    
    completion = groq_client.chat.completions.create(
        model=kwargs.get("model", "llama-3.3-70b-versatile"),
        messages=messages,
        temperature=kwargs.get("temperature", 0.1),
        max_tokens=kwargs.get("max_tokens", 2048)
    )
    return completion.choices[0].message.content

# ============================================
# EMBEDDING FUNCTION
# ============================================
embedding_func = EmbeddingFunc(
    embedding_dim=384,
    max_token_size=512,
    func=lambda texts: embedder.encode(texts, show_progress_bar=False).tolist(),
)

# ============================================
# INITIALIZE RAG-ANYTHING
# ============================================
@st.cache_resource
def initialize_rag():
    """Initialize RAG-Anything instance (cached)"""
    config = RAGAnythingConfig(
        working_dir="./rag_anything_storage",
        parser="mineru",
        parse_method="auto",
        enable_image_processing=False,  # No vision for now
        enable_table_processing=True,   # Process tables
        enable_equation_processing=True, # Process equations
    )
    
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=None,
        embedding_func=embedding_func,
    )
    
    return rag

# Get RAG instance
if st.session_state.rag_instance is None:
    with st.spinner("🔄 Initializing RAG-Anything..."):
        st.session_state.rag_instance = initialize_rag()

rag = st.session_state.rag_instance

# ============================================
# HELPER FUNCTIONS
# ============================================
def extract_pdf_bytes(pdf_bytes):
    """Extract text from PDF"""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for p in reader.pages:
        t = p.extract_text() or ""
        text += t + "\n"
    return text

def extract_xlsx_bytes(xlsx_bytes):
    """Extract text from Excel"""
    xls = pd.ExcelFile(io.BytesIO(xlsx_bytes))
    all_text = []
    
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        
        all_text.append(f"\n=== Sheet: {sheet_name} ===")
        all_text.append(f"Columns: {', '.join(str(col) for col in df.columns)}\n")
        
        for idx, row in df.iterrows():
            row_text = []
            for col in df.columns:
                value = row[col]
                if pd.notna(value):
                    row_text.append(f"{col}: {value}")
            
            if row_text:
                all_text.append(" | ".join(row_text))
                all_text.append("")
        
        all_text.append("\n")
    
    return "\n".join(all_text)

async def index_document_async(rag_instance, content, filename, file_type):
    """Index document using RAG-Anything"""
    # Create temporary file for RAG-Anything to process
    import tempfile
    import os
    
    temp_dir = tempfile.mkdtemp()
    
    if file_type == "pdf":
        temp_path = os.path.join(temp_dir, filename)
        with open(temp_path, 'wb') as f:
            f.write(content)
    elif file_type == "xlsx":
        temp_path = os.path.join(temp_dir, filename)
        with open(temp_path, 'wb') as f:
            f.write(content)
    
    try:
        # Use RAG-Anything to process document
        await rag_instance.process_document_complete(
            file_path=temp_path,
            output_dir="./output",
            parse_method="auto",
            display_stats=False,
        )
        success = True
    except Exception as e:
        st.error(f"Error indexing {filename}: {e}")
        success = False
    finally:
        # Cleanup temp file
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return success

async def query_async(rag_instance, question, model, max_tokens):
    """Query RAG-Anything with token tracking"""
    
    # Build custom prompt for Groq with token tracking
    start_time = time.time()
    
    try:
        # Query RAG-Anything (uses hybrid search + knowledge graph)
        result = await rag_instance.aquery(
            question,
            mode="hybrid",  # Best mode: combines local + global search
            vlm_enhanced=False,
        )
        
        elapsed = time.time() - start_time
        
        # For token tracking, we need to make a separate call to Groq
        # to get the actual tokens used
        messages = [{"role": "user", "content": f"Based on the following context, answer this question: {question}\n\nContext: {result}"}]
        
        completion = groq_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=max_tokens
        )
        
        usage = completion.usage
        
        return {
            "text": result,
            "tokens": {
                "prompt": usage.prompt_tokens,
                "completion": usage.completion_tokens,
                "total": usage.total_tokens
            },
            "finish_reason": completion.choices[0].finish_reason,
            "elapsed": elapsed
        }
    except Exception as e:
        return {
            "text": f"⚠️ Error: {str(e)}",
            "tokens": None,
            "finish_reason": "error",
            "elapsed": time.time() - start_time
        }

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.header("🤖 Model Settings")
    
    model_choice = st.selectbox(
        "Groq Model",
        [
            "llama-3.3-70b-versatile",
            "openai/gpt-oss-120b",
            "qwen/qwen3-32b",
            "llama-3.1-8b-instant",
        ],
        index=0,
        help="Select the model for query answering"
    )
    
    model_info = {
        "llama-3.3-70b-versatile": "70B | Best quality ⭐",
        "openai/gpt-oss-120b": "120B | Reasoning 🧠",
        "qwen/qwen3-32b": "32B | Balanced ⚖️",
        "llama-3.1-8b-instant": "8B | Fast ⚡",
    }
    
    st.caption(f"📊 {model_info.get(model_choice, 'Unknown')}")
    
    max_tokens = st.slider(
        "Max Output Tokens",
        min_value=500,
        max_value=4096,
        value=2048,
        step=256,
        help="Maximum length of the answer"
    )
    st.caption(f"📝 ~{int(max_tokens * 0.75)} words max")
    
    st.divider()
    
    st.header("📁 Upload Documents")
    pdf1 = st.file_uploader("PDF 1", type="pdf", key="pdf1")
    pdf2 = st.file_uploader("PDF 2", type="pdf", key="pdf2")
    excel = st.file_uploader("Excel file", type="xlsx", key="excel")
    
    clear_before_index = st.checkbox(
        "Clear previous documents before indexing",
        value=True,
        help="Removes old documents before adding new ones"
    )
    
    index_btn = st.button("Index Documents", use_container_width=True)
    
    if st.button("🗑️ Clear All Documents", use_container_width=True, type="secondary"):
        try:
            # Clear RAG-Anything storage
            import shutil
            shutil.rmtree("./rag_anything_storage", ignore_errors=True)
            st.session_state.indexed_files = []
            st.session_state.rag_instance = None
            st.success("✅ All documents cleared!")
            st.rerun()
        except Exception as e:
            st.error(f"Error clearing: {e}")
    
    if st.session_state.indexed_files:
        st.success("✅ Indexed files:")
        for f in st.session_state.indexed_files:
            st.text(f)
    
    st.divider()
    st.caption("🚀 Powered by RAG-Anything")
    st.caption("📊 Knowledge Graph Enabled")
    st.caption("⚡ Groq LLM Backend")

# ============================================
# INDEXING
# ============================================
if index_btn:
    st.subheader("Indexing documents with RAG-Anything…")
    
    if clear_before_index and st.session_state.indexed_files:
        with st.spinner("🗑️ Clearing previous documents..."):
            import shutil
            shutil.rmtree("./rag_anything_storage", ignore_errors=True)
            st.session_state.indexed_files = []
            st.session_state.rag_instance = None
            # Reinitialize
            st.session_state.rag_instance = initialize_rag()
            rag = st.session_state.rag_instance
            st.info("Cleared previous documents")
    
    file_names = []
    
    # Process PDF 1
    if pdf1:
        with st.spinner(f"📄 Indexing {pdf1.name}..."):
            pdf1_bytes = pdf1.read()
            success = asyncio.run(index_document_async(rag, pdf1_bytes, pdf1.name, "pdf"))
            if success:
                file_names.append(f"📄 {pdf1.name}")
    
    # Process PDF 2
    if pdf2:
        with st.spinner(f"📄 Indexing {pdf2.name}..."):
            pdf2_bytes = pdf2.read()
            success = asyncio.run(index_document_async(rag, pdf2_bytes, pdf2.name, "pdf"))
            if success:
                file_names.append(f"📄 {pdf2.name}")
    
    # Process Excel
    if excel:
        with st.spinner(f"📊 Indexing {excel.name}..."):
            excel_bytes = excel.read()
            success = asyncio.run(index_document_async(rag, excel_bytes, excel.name, "xlsx"))
            if success:
                file_names.append(f"📊 {excel.name}")
    
    if file_names:
        st.session_state.indexed_files = file_names
        st.success("✅ Documents indexed successfully with knowledge graph!")
        st.info("💡 RAG-Anything has built entity relationships for better retrieval")
    else:
        st.error("No documents uploaded!")

# ============================================
# MAIN AREA - Q&A
# ============================================
st.write("---")

if not st.session_state.indexed_files:
    st.info("👈 **Please upload and index documents in the sidebar to get started**")
else:
    st.header("💬 Ask Your Question")
    question = st.text_input(
        "Type your question here:",
        placeholder="e.g. Who teaches Math 101?",
        key="main_question"
    )
    ask_btn = st.button("🔍 Get Answer", type="primary", use_container_width=True)

    if ask_btn and question.strip():
        # Rate limit check
        current_time = time.time()
        time_since_last = current_time - st.session_state.last_request_time
        
        if time_since_last < 2:
            st.warning(f"⏳ Please wait {2 - time_since_last:.1f} seconds between questions")
            st.stop()
        
        st.session_state.last_request_time = current_time
        
        st.subheader("Answer")
        
        # Query RAG-Anything
        with st.spinner(f"🤔 Querying RAG-Anything with {model_choice}..."):
            response = asyncio.run(query_async(rag, question, model_choice, max_tokens))
        
        answer = response["text"]
        token_info = response.get("tokens")
        finish_reason = response.get("finish_reason")
        elapsed = response.get("elapsed", 0)
        
        # Warn if cut off
        if finish_reason == "length":
            st.error("⚠️ **Answer was cut off!** Increase max_tokens.")
        
        st.write("### 🧠 Answer:")
        st.write(answer)
        
        # Cost calculation
        cost_per_model = {
            "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
            "openai/gpt-oss-120b": {"input": 1.00, "output": 1.00},
            "qwen/qwen3-32b": {"input": 0.20, "output": 0.20},
            "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
        }
        
        if token_info:
            pricing = cost_per_model.get(model_choice, {"input": 0.20, "output": 0.20})
            input_cost = (token_info["prompt"] / 1_000_000) * pricing["input"]
            output_cost = (token_info["completion"] / 1_000_000) * pricing["output"]
            total_cost = input_cost + output_cost
        
        # Metrics
        st.write("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("⚡ Time", f"{elapsed:.2f}s")
        
        with col2:
            if token_info:
                st.metric("📥 Input Tokens", f"{token_info['prompt']:,}")
            else:
                st.metric("📥 Input", "N/A")
        
        with col3:
            if token_info:
                st.metric("📤 Output Tokens", f"{token_info['completion']:,}")
            else:
                st.metric("📤 Output", "N/A")
        
        with col4:
            if token_info:
                st.metric("💰 Cost", f"${total_cost:.6f}")
            else:
                st.metric("💰 Cost", "N/A")
        
        # Detailed breakdown
        if token_info:
            with st.expander("💡 Token & Cost Breakdown"):
                st.markdown(f"""
                **Model:** `{model_choice}`
                **RAG System:** RAG-Anything (Knowledge Graph + Vector Search)
                
                **Tokens:**
                - Input: {token_info['prompt']:,} tokens
                - Output: {token_info['completion']:,} tokens
                - **Total**: {token_info['total']:,} tokens
                
                **Cost:**
                - Input: ${input_cost:.6f} (${pricing['input']:.2f}/1M tokens)
                - Output: ${output_cost:.6f} (${pricing['output']:.2f}/1M tokens)
                - **Total**: ${total_cost:.6f}
                
                **Scale Estimates:**
                - 100 queries/day: ${total_cost * 100:.4f}/day
                - 1,000 queries/day: ${total_cost * 1000:.2f}/day
                
                **RAG-Anything Benefits:**
                - ✅ Knowledge graph relationships
                - ✅ Entity extraction
                - ✅ Hybrid retrieval (vector + graph)
                - ✅ Better context understanding
                """)