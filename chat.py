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


def query_llm_groq(question, context):
    """Use Groq API - SUPER FAST (500+ tokens/sec)"""
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
            model="llama-3.3-70b-versatile",  # Fast and accurate
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1000  # Increased to allow for longer lists
        )
        return completion.choices[0].message.content
    except Exception as e:
        error_msg = str(e)
        if "rate_limit" in error_msg.lower():
            return "⚠️ Rate limit reached. Please wait a moment and try again. (Groq free tier: 30 requests/minute)"
        else:
            return f"⚠️ Error generating answer: {error_msg}"


with st.sidebar:
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
        placeholder="e.g., What courses does Dr. Smith teach? or Who teaches Math 101?",
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
        with st.spinner("🤔 Generating answer..."):
            import time
            start = time.time()
            answer = query_llm_groq(question, context)
            elapsed = time.time() - start
            
        st.write("### 🧠 Answer:")
        st.write(answer)
        st.caption(f"⚡ Generated in {elapsed:.2f} seconds")