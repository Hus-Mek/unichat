# Agent Change Log and Lessons Learned

This file is used to track changes, errors, and lessons learned during development and deployment. It helps current and future agents avoid repeating mistakes and improve the workflow.

## Format
- **Date:** YYYY-MM-DD
- **Context:** Brief description of the situation or error
- **Action Taken:** What was done to resolve it
- **Lesson Learned:** What should be done differently next time

---

## Entries

- **Date:** 2026-03-25
  - **Context:** Railway deployment failed due to missing psycopg2 module (ModuleNotFoundError: No module named 'psycopg2').
  - **Action Taken:** Added `psycopg2-binary` to backend/requirements.txt and redeployed.
  - **Lesson Learned:** Always ensure all required database drivers are listed in requirements.txt before deploying to production environments.

- **Date:** 2026-03-25
  - **Context:** Railway deployment failed because DATABASE_URL used `${{Postgres.DATABASE_URL}}`, which defaults to the sync driver (`postgresql://`). Async SQLAlchemy requires the asyncpg driver.
  - **Action Taken:** Updated Railway environment variable to use `postgresql+asyncpg://${{Postgres.DATABASE_URL#postgresql://}}` so the asyncpg driver is used.
  - **Lesson Learned:** When using SQLAlchemy async engine, always ensure the DATABASE_URL uses the correct async driver prefix (`postgresql+asyncpg://`). For Railway, use string substitution to update the prefix.

- **Date:** 2026-03-25
  - **Context:** Need to deploy both backend (FastAPI) and frontend (Streamlit) on Railway, using a shared Postgres database.
  - **Action Taken:**
    1. Plan to deploy backend and frontend as separate Railway services.
    2. Backend: Set environment variables (DATABASE_URL with asyncpg, SECRET_KEY, etc.).
    3. Frontend: Set API_BASE_URL to backend's public URL (not Docker Compose internal address).
    4. Database: Use Railway's Postgres plugin and point backend's DATABASE_URL to it.
    5. CORS: Ensure backend allows requests from frontend's domain.
    6. Documented checklist and best practices for future deployments.
  - **Lesson Learned:**
    - In Railway, each service (backend, frontend, database) should be deployed separately.
    - Use public URLs for inter-service communication, not Docker Compose hostnames.
    - Always set CORS properly for frontend-backend communication.
    - Log all deployment steps and configs for reproducibility.

- **Date:** 2026-03-25
  - **Context:** Backend failed on Railway due to relative import errors (ModuleNotFoundError: No module named 'backend.models', etc.). Need to prepare for frontend deployment as well.
  - **Action Taken:**
    1. Converted all relative imports in backend/api/routers and api/deps.py to absolute imports for production compatibility.
    2. Confirmed backend structure and requirements for Railway deployment.
    3. Reviewed frontend requirements and Streamlit entrypoint for deployment.
  - **Lesson Learned:**
    - Always use absolute imports in production Python packages to avoid import errors in cloud environments.
    - Prepare frontend with correct API_BASE_URL pointing to backend's public Railway URL.
    - Log all code and deployment changes for future reference.

## Railway Deployment Checklist
- [ ] Deploy backend as a Railway service (set env vars, use asyncpg in DATABASE_URL)
- [ ] Deploy frontend as a Railway service (set API_BASE_URL to backend's public URL)
- [ ] Add Postgres plugin or deploy Postgres service
- [ ] Set backend DATABASE_URL to Railway Postgres connection string (with asyncpg prefix)
- [ ] Set CORS in backend to allow frontend domain
- [ ] Test end-to-end connectivity (frontend → backend → database)

