"""University Chatbot - Streamlit Frontend"""

import os

import httpx
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")


def api_request(method: str, path: str, **kwargs) -> httpx.Response:
    """Make an authenticated API request."""
    headers = kwargs.pop("headers", {})
    token = st.session_state.get("token")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return httpx.request(method, f"{API_BASE}{path}", headers=headers, timeout=60, **kwargs)


def login_page():
    """Login / Register page."""
    st.title("University Chatbot")
    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            resp = api_request("POST", "/auth/login", json={"email": email, "password": password})
            if resp.status_code == 200:
                data = resp.json()
                st.session_state["token"] = data["access_token"]
                st.session_state["user"] = data["user"]
                st.rerun()
            else:
                st.error(resp.json().get("detail", "Login failed"))

    with tab_register:
        reg_email = st.text_input("Email", key="reg_email")
        reg_user = st.text_input("Username", key="reg_user")
        reg_name = st.text_input("Full Name", key="reg_name")
        reg_pass = st.text_input("Password", type="password", key="reg_pass")
        reg_dept = st.text_input("Department", key="reg_dept")
        if st.button("Register"):
            resp = api_request("POST", "/auth/register", json={
                "email": reg_email, "username": reg_user,
                "password": reg_pass, "full_name": reg_name,
                "department": reg_dept,
            })
            if resp.status_code == 201:
                st.success("Account created! Please log in.")
            else:
                st.error(resp.json().get("detail", "Registration failed"))


def chat_page():
    """Main chat interface."""
    st.title("Chat")
    user = st.session_state["user"]

    # Sidebar: collection and model selection
    with st.sidebar:
        st.subheader(f"Logged in as {user['username']}")
        st.caption(f"Access: {user['access_level']}")
        if st.button("Logout"):
            st.session_state.clear()
            st.rerun()

        st.divider()

        # Load collections
        resp = api_request("GET", "/collections/")
        collections = resp.json() if resp.status_code == 200 else []
        col_names = ["All Collections"] + [c["name"] for c in collections]
        selected_col = st.selectbox("Collection", col_names)

        # Model selection
        resp_models = api_request("GET", "/health/models")
        models_data = resp_models.json() if resp_models.status_code == 200 else {}
        model_list = list(models_data.get("models", {}).keys())
        selected_model = st.selectbox("Model", model_list) if model_list else None

    # Chat history in session
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = None

    # Display messages
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.caption(f"{s['file_name']} ({s['access_level']}) - {s['chunks']} chunks")
            if msg.get("cost"):
                st.caption(f"Cost: ${msg['cost']:.6f} | Tokens: {msg.get('tokens_total', 'N/A')} | Time: {msg.get('time_ms', 'N/A')}ms")

    # Chat input
    if question := st.chat_input("Ask a question..."):
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        # Build request
        collection_ids = None
        if selected_col != "All Collections":
            matched = [c for c in collections if c["name"] == selected_col]
            if matched:
                collection_ids = [matched[0]["id"]]

        payload = {
            "question": question,
            "collection_ids": collection_ids,
            "model": selected_model,
            "session_id": st.session_state.get("session_id"),
        }

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                resp = api_request("POST", "/chat/query", json=payload)

            if resp.status_code == 200:
                data = resp.json()
                st.write(data["answer"])
                st.session_state["session_id"] = data["session_id"]

                sources = data.get("sources", [])
                if sources:
                    with st.expander("Sources"):
                        for s in sources:
                            st.caption(f"{s['file_name']} ({s['access_level']}) - {s['chunks']} chunks")

                tokens = data.get("tokens")
                cost = data.get("cost_usd", 0)
                time_ms = data.get("response_time_ms", 0)
                tokens_total = tokens["total"] if tokens else "N/A"
                st.caption(f"Cost: ${cost:.6f} | Tokens: {tokens_total} | Time: {time_ms}ms")

                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": data["answer"],
                    "sources": sources,
                    "cost": cost,
                    "tokens_total": tokens_total,
                    "time_ms": time_ms,
                })
            else:
                error = resp.json().get("detail", "Query failed")
                st.error(error)


def upload_page():
    """Document upload page."""
    st.title("Upload Documents")

    resp = api_request("GET", "/collections/")
    collections = resp.json() if resp.status_code == 200 else []

    if not collections:
        st.warning("No collections available. An admin must create one first.")
        return

    col_options = {c["name"]: c["id"] for c in collections}
    selected = st.selectbox("Target Collection", list(col_options.keys()))
    access_level = st.selectbox("Access Level", ["public", "student", "faculty", "admin_staff", "executive_board"])

    files = st.file_uploader(
        "Upload documents (PDF, Excel, Word)",
        type=["pdf", "xlsx", "xls", "docx"],
        accept_multiple_files=True,
    )

    if st.button("Upload & Index") and files:
        collection_id = col_options[selected]
        for file in files:
            with st.spinner(f"Processing {file.name}..."):
                resp = api_request(
                    "POST",
                    "/documents/upload",
                    files={"file": (file.name, file.getvalue(), file.type)},
                    data={"collection_id": str(collection_id), "access_level": access_level},
                )
                if resp.status_code == 201:
                    data = resp.json()
                    st.success(f"{file.name}: indexed ({data['chunk_count']} chunks)")
                else:
                    st.error(f"{file.name}: {resp.json().get('detail', 'Failed')}")

    # List existing documents
    st.divider()
    st.subheader("Existing Documents")
    resp = api_request("GET", "/documents/")
    if resp.status_code == 200:
        docs = resp.json().get("documents", [])
        if docs:
            for doc in docs:
                col = st.columns([3, 1, 1, 1])
                col[0].write(doc["file_name"])
                col[1].write(doc["access_level"])
                col[2].write(f"{doc['chunk_count']} chunks")
                col[3].write(doc["status"])
        else:
            st.info("No documents uploaded yet.")


def admin_page():
    """Admin panel for user management and system stats."""
    st.title("Admin Panel")
    user = st.session_state["user"]

    if user["access_level"] not in ("admin_staff", "executive_board"):
        st.error("Access denied. Admin staff or higher required.")
        return

    tab_stats, tab_users, tab_audit, tab_report = st.tabs(
        ["Statistics", "Users", "Audit Logs", "Generate Report"]
    )

    with tab_stats:
        resp = api_request("GET", "/admin/stats")
        if resp.status_code == 200:
            stats = resp.json()
            cols = st.columns(5)
            cols[0].metric("Users", stats["total_users"])
            cols[1].metric("Documents", stats["total_documents"])
            cols[2].metric("Collections", stats["total_collections"])
            cols[3].metric("Queries", stats["total_queries"])
            cols[4].metric("Chunks", stats["total_chunks"])

    with tab_users:
        resp = api_request("GET", "/admin/users")
        if resp.status_code == 200:
            users = resp.json()
            for u in users:
                with st.expander(f"{u['username']} ({u['email']})"):
                    st.write(f"Access: {u['access_level']}")
                    st.write(f"Department: {u.get('department', 'N/A')}")
                    st.write(f"Active: {u['is_active']}")
                    new_level = st.selectbox(
                        "Change access level",
                        ["public", "student", "faculty", "admin_staff", "executive_board"],
                        index=["public", "student", "faculty", "admin_staff", "executive_board"].index(u["access_level"]),
                        key=f"level_{u['id']}",
                    )
                    if st.button("Update", key=f"update_{u['id']}"):
                        resp = api_request("PUT", f"/admin/users/{u['id']}", json={"access_level": new_level})
                        if resp.status_code == 200:
                            st.success("Updated")
                            st.rerun()

    with tab_audit:
        resp = api_request("GET", "/admin/audit-logs?limit=50")
        if resp.status_code == 200:
            logs = resp.json()
            for log in logs:
                st.text(f"[{log['created_at']}] {log['action']} - {log.get('resource_type', '')} - User: {log.get('user_id', 'N/A')}")

    with tab_report:
        st.subheader("Generate University Proposal")
        uni_name = st.text_input("University Name", "University")
        daily_queries = st.number_input("Expected Daily Queries", value=1000, min_value=100)
        deployment = st.selectbox("Preferred Deployment", ["onprem", "cloud", "hybrid", "undecided"])

        if st.button("Generate PowerPoint"):
            with st.spinner("Generating presentation..."):
                resp = api_request("POST", "/admin/generate-report", json={
                    "university_name": uni_name,
                    "expected_daily_queries": daily_queries,
                    "preferred_deployment": deployment if deployment != "undecided" else None,
                })
                if resp.status_code == 200:
                    st.download_button(
                        "Download Presentation",
                        data=resp.content,
                        file_name="university_chatbot_proposal.pptx",
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    )
                else:
                    st.error("Failed to generate report")


def main():
    st.set_page_config(page_title="University Chatbot", layout="wide")

    if "token" not in st.session_state:
        login_page()
        return

    # Navigation
    with st.sidebar:
        page = st.radio("Navigate", ["Chat", "Upload Documents", "Admin"])

    if page == "Chat":
        chat_page()
    elif page == "Upload Documents":
        upload_page()
    elif page == "Admin":
        admin_page()


if __name__ == "__main__":
    main()
