"""
Streamlit Application
Main UI for University RAG Assistant
"""

import streamlit as st
import time
from pathlib import Path
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our modules using absolute imports
from src.config import Config
from src.rag_engine import RAGEngine
from src.llm_client import LLMClient
from src.access_control import AccessController

# ============================================
# SESSION STATE
# ============================================
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "last_request_time" not in st.session_state:
    st.session_state.last_request_time = 0
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None
if "llm_client" not in st.session_state:
    st.session_state.llm_client = None
if "access_controller" not in st.session_state:
    st.session_state.access_controller = None

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="University Assistant",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 University RAG Assistant")
st.caption("Professional RAG system with 3-level access control")

# ============================================
# INITIALIZATION
# ============================================
# Get API key
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    st.error("⚠️ GROQ_API_KEY not found in secrets. Please configure it.")
    st.stop()

# Initialize components (cached in session state)
if st.session_state.rag_engine is None:
    with st.spinner("Initializing RAG engine..."):
        st.session_state.rag_engine = RAGEngine()
        st.session_state.llm_client = LLMClient(GROQ_API_KEY)
        st.session_state.access_controller = AccessController()

rag_engine = st.session_state.rag_engine
llm_client = st.session_state.llm_client
access_controller = st.session_state.access_controller

# ============================================
# SIDEBAR - USER CONTEXT & SETTINGS
# ============================================
with st.sidebar:
    st.header("👤 User Context")
    
    # Access level selection
    user_access_level = st.selectbox(
        "Access Level",
        list(Config.ACCESS_LEVELS.keys()),
        format_func=lambda x: Config.ACCESS_LEVELS[x],
        help="Select your access level"
    )
    
    # User ID for personal documents
    user_id = st.text_input(
        "User ID (optional)",
        placeholder="e.g., stu_12345",
        help="For accessing personal documents"
    )
    
    # Show permissions
    st.caption("**You can access:**")
    st.caption(access_controller.get_permissions_text(user_access_level))
    
    st.divider()
    
    # Model settings
    st.header("🤖 Model Settings")
    
    model_choice = st.selectbox(
        "Groq Model",
        Config.get_model_list(),
        help="Choose the AI model"
    )
    
    model_config = Config.get_model_config(model_choice)
    st.caption(f"📊 {model_config.description}")
    
    max_tokens = st.slider(
        "Max Output Tokens",
        min_value=500,
        max_value=4096,
        value=2048,
        step=256,
        help="Maximum answer length"
    )
    
    st.divider()
    
    # Document upload section
    st.header("📁 Upload Documents")
    
    # File uploaders
    uploaded_files = st.file_uploader(
        "Choose Documents",
        type=['pdf', 'xlsx', 'xls'],
        accept_multiple_files=True,
        help="Upload PDF or Excel files"
    )
    
    # Store document settings in session state
    if "doc_settings" not in st.session_state:
        st.session_state.doc_settings = {}
    
    # If files are uploaded, show settings for each
    if uploaded_files:
        st.write("---")
        st.subheader("📋 Configure Each Document")
        
        for idx, uploaded_file in enumerate(uploaded_files):
            file_key = f"{uploaded_file.name}_{uploaded_file.size}"
            
            # Initialize settings for this file if not exists
            if file_key not in st.session_state.doc_settings:
                st.session_state.doc_settings[file_key] = {
                    "access_level": "public",
                    "owner": ""
                }
            
            with st.expander(f"📄 {uploaded_file.name}", expanded=True):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    access_level = st.selectbox(
                        "Access Level",
                        list(Config.ACCESS_LEVELS.keys()),
                        format_func=lambda x: Config.ACCESS_LEVELS[x],
                        key=f"access_{file_key}",
                        index=list(Config.ACCESS_LEVELS.keys()).index(
                            st.session_state.doc_settings[file_key]["access_level"]
                        )
                    )
                    st.session_state.doc_settings[file_key]["access_level"] = access_level
                
                with col2:
                    owner = st.text_input(
                        "Owner (optional)",
                        placeholder="stu_12345",
                        key=f"owner_{file_key}",
                        value=st.session_state.doc_settings[file_key]["owner"]
                    )
                    st.session_state.doc_settings[file_key]["owner"] = owner
                
                # Show info
                st.caption(f"📊 Size: {uploaded_file.size / 1024:.1f} KB | Type: {uploaded_file.type}")
        
        st.write("---")
    
    clear_before_index = st.checkbox(
        "Clear previous documents before indexing",
        value=True,
        help="Remove old documents before indexing new ones"
    )
    
    index_btn = st.button(
        "📚 Index All Documents",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files
    )
    
    # Clear database button
    if st.button("🗑️ Clear All Documents", use_container_width=True):
        result = rag_engine.clear_database()
        if result["success"]:
            st.session_state.indexed_files = []
            st.success(f"✅ Cleared {result.get('deleted', 0)} chunks")
            st.rerun()
        else:
            st.error(f"Error: {result.get('error')}")
    
    # Show indexed files
    if st.session_state.indexed_files:
        st.success("✅ Indexed:")
        for f in st.session_state.indexed_files:
            st.text(f)
    
    # Database stats
    stats = rag_engine.get_stats()
    if stats["success"]:
        st.divider()
        st.caption(f"📊 Total chunks: {stats['total_chunks']}")

# ============================================
# DOCUMENT INDEXING
# ============================================
if index_btn and uploaded_files:
    st.subheader("📚 Indexing Documents")
    
    # Clear if requested
    if clear_before_index:
        result = rag_engine.clear_database()
        if result["success"] and result.get("deleted", 0) > 0:
            st.info(f"🗑️ Cleared {result['deleted']} previous chunks")
    
    # Index each file with its individual settings
    indexed_files = []
    progress_bar = st.progress(0)
    
    for idx, uploaded_file in enumerate(uploaded_files):
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        settings = st.session_state.doc_settings.get(file_key, {
            "access_level": "public",
            "owner": ""
        })
        
        with st.spinner(f"Indexing {uploaded_file.name}..."):
            file_bytes = uploaded_file.read()
            file_extension = Path(uploaded_file.name).suffix
            
            result = rag_engine.index_document(
                file_bytes=file_bytes,
                file_name=uploaded_file.name,
                file_extension=file_extension,
                access_level=settings["access_level"],
                owner=settings["owner"] if settings["owner"] else None
            )
            
            if result["success"]:
                access_label = Config.ACCESS_LEVELS[settings["access_level"]]
                file_icon = "📄" if file_extension == ".pdf" else "📊"
                
                # Include owner in display if set
                owner_text = f" | Owner: {settings['owner']}" if settings["owner"] else ""
                indexed_files.append(
                    f"{file_icon} {result['file_name']} ({access_label}{owner_text}) - {result['chunks']} chunks"
                )
                st.success(f"✅ {result['file_name']}: {result['chunks']} chunks indexed")
            else:
                st.error(f"❌ Failed: {uploaded_file.name} - {result['error']}")
        
        # Update progress
        progress_bar.progress((idx + 1) / len(uploaded_files))
    
    # Update session state and clear settings
    st.session_state.indexed_files = indexed_files
    st.session_state.doc_settings = {}
    
    st.success(f"🎉 Successfully indexed {len(indexed_files)} document(s)!")
    st.balloons()

# ============================================
# MAIN AREA - QUESTION & ANSWER
# ============================================
st.write("---")

if not st.session_state.indexed_files:
    st.info("👈 **Please upload and index documents in the sidebar to get started**")
else:
    st.header("💬 Ask Your Question")
    
    question = st.text_input(
        "Type your question here:",
        placeholder="e.g., What courses are offered?",
        key="main_question"
    )
    
    ask_btn = st.button("🔍 Get Answer", type="primary", use_container_width=True)
    
    if ask_btn and question.strip():
        # Rate limit check
        current_time = time.time()
        time_since_last = current_time - st.session_state.last_request_time
        
        if time_since_last < Config.MIN_REQUEST_INTERVAL:
            st.warning(f"⏳ Please wait {Config.MIN_REQUEST_INTERVAL - time_since_last:.1f}s")
            st.stop()
        
        st.session_state.last_request_time = current_time
        
        # Show user context
        st.info(f"🔍 Searching as: **{Config.ACCESS_LEVELS[user_access_level]}** user")
        if user_id:
            st.caption(f"Including documents owned by: {user_id}")
        
        # Retrieve documents
        with st.spinner("Retrieving relevant documents..."):
            retrieval_result = rag_engine.query(
                question=question,
                user_level=user_access_level,
                user_id=user_id if user_id else None
            )
        
        if not retrieval_result["success"]:
            st.error(f"Error retrieving documents: {retrieval_result.get('error')}")
            st.stop()
        
        if not retrieval_result["documents"]:
            st.warning(f"⚠️ No accessible documents found for: **{Config.ACCESS_LEVELS[user_access_level]}**")
            st.info("💡 Try uploading documents with your access level or changing your user level.")
            st.stop()
        
        st.success(f"✓ Found {retrieval_result['count']} relevant chunks")
        
        # Query LLM
        with st.spinner(f"🤔 Generating answer with {model_choice}..."):
            start = time.time()
            llm_response = llm_client.query(
                question=question,
                context=retrieval_result["context"],
                model=model_choice,
                max_tokens=max_tokens
            )
            elapsed = time.time() - start
        
        if not llm_response["success"]:
            st.error(f"Error generating answer: {llm_response.get('error')}")
            st.stop()
        
        # Display answer
        st.write("### 🧠 Answer:")
        st.write(llm_response["text"])
        
        # Warning if cut off
        if llm_response["finish_reason"] == "length":
            st.warning("⚠️ Answer was cut off! Try increasing max_tokens.")
        
        # Calculate cost
        cost_info = llm_client.calculate_cost(
            llm_response["tokens"],
            model_choice
        )
        
        # Display metrics
        st.write("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("⏱ Time", f"{elapsed:.2f}s")
        
        with col2:
            if llm_response["tokens"]:
                st.metric("📥 Input", f"{llm_response['tokens']['prompt']:,}")
        
        with col3:
            if llm_response["tokens"]:
                st.metric("📤 Output", f"{llm_response['tokens']['completion']:,}")
        
        # Detailed breakdown (with unique key to force update)
        if llm_response["tokens"]:
            # Use question hash + timestamp to force refresh
            expander_key = f"breakdown_{hash(question)}_{int(time.time() * 1000)}"
            with st.expander("📊 Detailed Breakdown", expanded=False):
                # Show access levels of retrieved documents
                retrieved_metadatas = retrieval_result.get("metadatas", [])
                sources_detail = retrieval_result.get("sources_detail", {})
                
                if retrieved_metadatas:
                    access_levels_found = {}
                    for meta in retrieved_metadatas:
                        level = meta.get("access_level", "unknown")
                        owner = meta.get("owner", "")
                        key = f"{level}"
                        if owner:
                            key += f" (owner: {owner})"
                        access_levels_found[key] = access_levels_found.get(key, 0) + 1
                    
                    retrieved_docs_info = "\n".join([f"- {k}: {v} chunks" for k, v in access_levels_found.items()])
                else:
                    retrieved_docs_info = "⚠️ No metadata available (old version - please re-deploy)"
                
                # Detailed source breakdown
                if sources_detail:
                    sources_breakdown = "\n".join([
                        f"- **{source}** ({info['access_level']}): {info['count']} chunks" 
                        for source, info in sources_detail.items()
                    ])
                else:
                    sources_breakdown = "\n".join(['- ' + src for src in retrieval_result['sources']])
                
                st.markdown(f"""
                **Model:** `{model_choice}` - {model_config.description}
                
                **Your Access Level:** {Config.ACCESS_LEVELS[user_access_level]}
                {f"**Your User ID:** {user_id}" if user_id else ""}
                
                **Retrieved Documents by Access Level:**
                {retrieved_docs_info}
                
                **Retrieved Documents by Source:**
                {sources_breakdown}
                
                **Tokens:**
                - Input: {llm_response['tokens']['prompt']:,}
                - Output: {llm_response['tokens']['completion']:,}
                - Total: {llm_response['tokens']['total']:,}
                """)