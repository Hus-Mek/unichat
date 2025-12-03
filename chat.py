import streamlit as st
import pandas as pd
import json
import io
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq  # pip install groq

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_request_time" not in st.session_state:
    st.session_state.last_request_time = 0

st.set_page_config(page_title="Uni Assistant Demo", layout="wide")
st.title("Chatbot Demo")

EMBED_MODEL = "all-MiniLM-L6-v2"
CHROMA_DIR = "demo_store"

# Groq API setup - Uses Streamlit Secrets
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    st.error("⚠️ API key not configured. Please contact the administrator.")
    st.stop()

groq_client = Groq(api_key=GROQ_API_KEY)


@st.cache_resource
def get_embedder():
    return SentenceTransformer(EMBED_MODEL)

embedder = get_embedder()

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection("uni_docs")


def extract_pdf_bytes(pdf_bytes):
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for p in reader.pages:
        t = p.extract_text() or ""
        text += t + "\n"
    return text


def extract_xlsx_bytes(xlsx_bytes):
    """Extract text from Excel - IMPROVED to keep rows together better"""
    xls = pd.ExcelFile(io.BytesIO(xlsx_bytes))
    all_text = []
    
    # Process each sheet
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        
        # Add sheet header with more context
        all_text.append(f"\n=== Sheet: {sheet_name} ===")
        all_text.append(f"Columns: {', '.join(str(col) for col in df.columns)}\n")
        
        # Convert each row to a complete record - keep them together
        for idx, row in df.iterrows():
            row_text = []
            for col in df.columns:
                value = row[col]
                if pd.notna(value):  # Skip empty cells
                    row_text.append(f"{col}: {value}")
            
            if row_text:  # Only add non-empty rows
                # Each row on its own line for better chunking
                all_text.append(" | ".join(row_text))
                all_text.append("")  # Empty line between rows
        
        all_text.append("\n")
    
    return "\n".join(all_text)


def chunk_text(text, max_length=1200):
    """Slightly larger chunks to keep Excel rows together"""
    parts = []
    lines = text.split("\n")
    buf = ""
    
    for line in lines:
        line = line.strip()
        if not line:  # Skip empty lines
            continue
            
        # If adding this line exceeds max_length and we have content, save chunk
        if len(buf) + len(line) > max_length and buf:
            parts.append(buf.strip())
            buf = line + "\n"
        else:
            buf += line + "\n"
    
    # Don't forget the last chunk
    if buf.strip():
        parts.append(buf.strip())
    
    return parts


def query_llm_groq(question, context, model="llama-3.1-70b-versatile", max_tokens=2048):
    """Use Groq API - Returns response with token usage"""
    prompt = f"""You are a university assistant. 
Answer the question using ONLY the context below.
If the answer is not in the context, say you don't know.

IMPORTANT INSTRUCTIONS:
1. If the question asks for a list, provide ALL results found in the context.
2. Look carefully at ALL information - relationships work both ways (e.g., if Prof X teaches Course Y, then Course Y is taught by Prof X).
3. Search through the ENTIRE context before concluding information is unavailable.

CONTEXT:
\"\"\"{context}\"\"\"

QUESTION:
{question}"""

    try:
        completion = groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=max_tokens  # Now uses the parameter
        )
        
        # Extract token usage
        usage = completion.usage
        response_text = completion.choices[0].message.content
        
        # Check if response was cut off
        finish_reason = completion.choices[0].finish_reason
        
        # Return dict with text and usage info
        return {
            "text": response_text,
            "tokens": {
                "prompt": usage.prompt_tokens,
                "completion": usage.completion_tokens,
                "total": usage.total_tokens
            },
            "finish_reason": finish_reason
        }
    except Exception as e:
        error_msg = str(e)
        if "rate_limit" in error_msg.lower():
            return {"text": "⚠️ Rate limit reached. Please wait a moment and try again.", "tokens": None, "finish_reason": "error"}
        else:
            return {"text": f"⚠️ Error: {error_msg}", "tokens": None, "finish_reason": "error"}


with st.sidebar:
    st.header("🤖 Model Settings")
    
    # Model selection - expanded with preview models
    model_choice = st.selectbox(
        "Groq Model",
        [
            # Production models (stable)
            "llama-3.3-70b-versatile",      # 70B - Best quality
            "openai/gpt-oss-120b",          # 120B - Largest
            "qwen/qwen3-32b",               # 32B - Balanced
            "llama-3.1-8b-instant",         # 8B - Fast
            
         
        ],
        index=0,  # Default to 3.3 70B
        help="Production = stable | Preview = experimental but faster"
    )
    
    # Show model info with more details
    model_info = {
        # Production
        "llama-3.3-70b-versatile": "70B | Production | Best quality ",
        "openai/gpt-oss-120b": "120B | Production | Reasoning + Search ",
        "qwen/qwen3-32b": "32B | Production | Balanced ",
        "llama-3.1-8b-instant": "8B | Production | Fast ",
        
    }
    
    st.caption(f"📊 {model_info.get(model_choice, 'Unknown')}")
    
    # Warn if preview model selected
    if "preview" in model_choice or "deepseek" in model_choice:
        st.warning("⚠️ Preview model: May be less accurate or deprecated soon")
    
    # Max tokens setting
    max_tokens = st.slider(
        "Max Output Tokens",
        min_value=500,
        max_value=4096,
        value=2048,
        step=256,
        help="Maximum length of the answer. Increase if answers are cut off."
    )
    st.caption(f"📝 Allows answers up to ~{int(max_tokens * 0.75)} words")
    
    
    
    st.divider()
    
    st.header("📁 Upload Documents")
    pdf1 = st.file_uploader("PDF 1", type="pdf")
    pdf2 = st.file_uploader("PDF 2", type="pdf")
    excel = st.file_uploader("Excel file", type="xlsx")
    
    # Option to clear database before indexing
    clear_before_index = st.checkbox("Clear previous documents before indexing", value=True, 
                                      help="Recommended: removes old documents before adding new ones")
    
    index_btn = st.button("Index Documents", use_container_width=True)
    
    # Manual clear button
    if st.button("🗑️ Clear All Documents", use_container_width=True, type="secondary"):
        try:
            # Delete all documents from collection
            all_ids = collection.get()["ids"]
            if all_ids:
                collection.delete(ids=all_ids)
                st.session_state.indexed_files = []
                st.success("✅ All documents cleared!")
                st.rerun()
            else:
                st.info("No documents to clear")
        except Exception as e:
            st.error(f"Error clearing: {e}")
    
    # Show indexed files
    if st.session_state.indexed_files:
        st.success("✅ Indexed files:")
        for f in st.session_state.indexed_files:
            st.text(f)
    
    st.divider()
    st.caption("ℹ️ Using Groq free tier: 30 requests/minute")
    st.caption("💡 Wait 2 seconds between questions")


    if index_btn:
        st.subheader("Indexing documents…")
        
        # Clear existing documents if checkbox is selected
        if clear_before_index:
            try:
                all_ids = collection.get()["ids"]
                if all_ids:
                    collection.delete(ids=all_ids)
                    st.info("🗑️ Cleared previous documents")
            except Exception as e:
                st.warning(f"Could not clear previous documents: {e}")
        
        all_chunks = []
        file_names = []

        if pdf1:
            pdf1_bytes = pdf1.read()  
            text = extract_pdf_bytes(pdf1_bytes)  
            chunks = chunk_text(text)
            for i, c in enumerate(chunks):
                all_chunks.append({"id": f"pdf1_{i}", "text": c})
            file_names.append(f"📄 {pdf1.name}")

        if pdf2:
            pdf2_bytes = pdf2.read()  
            text = extract_pdf_bytes(pdf2_bytes)  
            chunks = chunk_text(text)
            for i, c in enumerate(chunks):
                all_chunks.append({"id": f"pdf2_{i}", "text": c})
            file_names.append(f"📄 {pdf2.name}")
        
        if excel:
            excel_bytes = excel.read()
            text = extract_xlsx_bytes(excel_bytes)
            st.write("### DEBUG: Excel extracted text")
            st.write(f"Length: {len(text)} characters")
            st.text_area("First 500 chars:", text[:500], height=200)
            chunks = chunk_text(text)
            st.write(f"Created {len(chunks)} chunks from Excel")
            for i, c in enumerate(chunks):
                all_chunks.append({"id": f"xlsx_{i}", "text": c})
            file_names.append(f"📊 {excel.name}")
        
        if not all_chunks:
            st.error("No documents uploaded!")
            st.stop()

        # Embed + Insert
        st.write("Embedding…")
        texts = [c["text"] for c in all_chunks]
        embeddings = embedder.encode(texts, show_progress_bar=True).tolist()
        ids = [c["id"] for c in all_chunks]

        collection.add(ids=ids, documents=texts, embeddings=embeddings)
        st.session_state.indexed_files = file_names
        st.success("Documents indexed successfully!")

# Main area - Question and Answer
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
        # Check rate limit cooldown (2 seconds between requests)
        import time
        current_time = time.time()
        time_since_last = current_time - st.session_state.last_request_time
        
        if time_since_last < 2:
            st.warning(f"⏳ Please wait {2 - time_since_last:.1f} seconds between questions to avoid rate limits.")
            st.stop()
        
        st.session_state.last_request_time = current_time
        
        st.subheader("Answer")

        # Retrieve MORE chunks to handle bidirectional queries better
        # e.g., "courses by Prof X" vs "who teaches course Y"
        q_emb = embedder.encode([question]).tolist()[0]
        result = collection.query(query_embeddings=[q_emb], n_results=15)

        retrieved_docs = result["documents"][0]
        
        # # DEBUG - Uncomment to see retrieved chunks
        # st.write(f"**DEBUG: Found {len(retrieved_docs)} chunks**")
        # question_lower = question.lower()
        # if "course" in question_lower or "class" in question_lower:
        #     st.info("💡 Tip: Check if the course/prof name appears in the retrieved chunks below")
        # for i, doc in enumerate(retrieved_docs):
        #     st.write(f"**Chunk {i+1}:**")
        #     st.text_area(f"Content {i+1}", doc, height=150, key=f"chunk_{i}")
        
        context = "\n\n".join(retrieved_docs)

        # # DEBUG - Uncomment to view full context
        # with st.expander("📚 View Retrieved Context"):
        #     st.code(context, language="text")

        # Show that answer is being generated
        with st.spinner(f"🤔 Generating answer with {model_choice}..."):
            import time
            start = time.time()
            response = query_llm_groq(question, context, model=model_choice, max_tokens=max_tokens)
            elapsed = time.time() - start
        
        # Extract answer and tokens
        answer = response["text"]
        token_info = response.get("tokens")
        finish_reason = response.get("finish_reason")
        
        # Warn if response was cut off
        if finish_reason == "length":
            st.error("⚠️ **Answer was cut off!** The response hit the token limit. Try asking for a shorter answer or increase max_tokens in the code.")
            
        st.write("### 🧠 Answer:")
        st.write(answer)
        
        # Cost calculation (Groq pricing as of 2024)
        cost_per_model = {
            "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},  # per 1M tokens
            "openai/gpt-oss-120b": {"input": 1.00, "output": 1.00},
            "qwen/qwen3-32b": {"input": 0.20, "output": 0.20},
            "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
            "llama-3.2-3b-preview": {"input": 0.05, "output": 0.08},
            "llama-3.2-1b-preview": {"input": 0.04, "output": 0.04},
            "deepseek-r1-distill-qwen-32b": {"input": 0.20, "output": 0.20},
            "deepseek-r1-distill-llama-70b": {"input": 0.59, "output": 0.79},
        }
        
        # Calculate cost if we have token info
        if token_info:
            pricing = cost_per_model.get(model_choice, {"input": 0.20, "output": 0.20})
            input_cost = (token_info["prompt"] / 1_000_000) * pricing["input"]
            output_cost = (token_info["completion"] / 1_000_000) * pricing["output"]
            total_cost = input_cost + output_cost
        
        # Show metrics
        st.write("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(" Time", f"{elapsed:.2f}s")
        
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
                st.metric(" Cost", f"${total_cost:.6f}")
            else:
                st.metric(" Cost", "N/A")
        
        # Show detailed breakdown in expander
        if token_info:
            with st.expander(" Token & Cost Breakdown"):
                st.markdown(f"""
                **Model:** `{model_choice}`
                
                **Tokens:**
                - Input (Context + Question): {token_info['prompt']:,} tokens
                - Output (Answer): {token_info['completion']:,} tokens
                - **Total**: {token_info['total']:,} tokens
                
                **Cost (Groq Pricing):**
                - Input: ${input_cost:.6f} (${pricing['input']:.2f} per 1M tokens)
                - Output: ${output_cost:.6f} (${pricing['output']:.2f} per 1M tokens)
                - **Total**: ${total_cost:.6f}
                
                **Free Tier Limits:**
                - ✅ 30 requests per minute
                - ✅ 14,400 requests per day
                - ✅ Free tier has no token limits!
                
                **Estimated costs at scale:**
                - 100 queries/day: ${total_cost * 100:.4f}/day
                - 1,000 queries/day: ${total_cost * 1000:.2f}/day
                - 10,000 queries/day: ${total_cost * 10000:.2f}/day
                """)