# 🧠 PsycheMind — AI-Powered Psychology Assistant

A Retrieval-Augmented Generation (RAG) product with a professional Streamlit
frontend. Same pipeline as before, now with a proper brand, a sidebar
dashboard, two knowledge-source modes, and a real conversation view.

**Pipeline:** PDF → clean → chunk → embed (`all-MiniLM-L6-v2`) → Chroma →
retrieve top-10 → rerank (cross-encoder `ms-marco-MiniLM-L-6-v2`) → top-4 →
Gemini generates a cited answer.

## What's new vs. the previous version

- **Branding:** PsycheMind name, logo (Concept 1 — brain + neural-network
  icon), purple/navy palette, badges, header, footer.
- **Sidebar dashboard:** knowledge-source switch, file uploader, pipeline
  status, model info, About section.
- **Two knowledge sources:**
  - 📖 **Psychology Book** — the existing persistent Psychology 2e database
    (unchanged).
  - 📂 **My Uploaded Document** — upload a PDF/TXT/DOCX; it's extracted,
    chunked, and embedded into its own temporary, in-memory vector store.
    Answers in this mode are grounded **only** in that document.
- **Conversation view:** multiple questions in one session, each with its
  own retrieval breakdown and sources, via `st.session_state`.
- **Retrieval Process panel:** a collapsible, plain-language breakdown of
  what happened for each answer (retrieved → reranked → context built →
  answered) — no raw logs.
- Friendly error messages everywhere instead of raw Python tracebacks.

The retrieval → reranking → citation logic itself (top-10 → cross-encoder →
top-4 → prompt rules) is **unchanged**.

## Files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit app — UI, session state, RAG orchestration |
| `vectorstore_utils.py` | Downloads the Psychology 2e PDF, builds/loads its persistent Chroma store |
| `utils/document_processing.py` | Extracts/chunks/embeds an uploaded PDF/TXT/DOCX into a temporary vector store |
| `assets/psychemind_logo.png`, `assets/logo_b64.txt` | The logo (transparent PNG + its base64 copy, used inline in the app so no external image hosting is needed) |
| `requirements.txt` | Python dependencies |
| `.streamlit/config.toml` | Native widget theme (matches the brand palette) |
| `.streamlit/secrets.toml.example` | Shows the secret key the app needs |

The Psychology 2e vector store is **not** stored in the repo — the app
builds `chroma_db/` automatically on first run (downloads the PDF, chunks,
embeds), then reuses it. If you already have a `chroma_db/` folder from a
previous run (e.g. from Colab), drop it in the project root next to
`app.py` and the app will load it directly instead of rebuilding it.

## Run locally

```bash
pip install -r requirements.txt
export GOOGLE_API_KEY="your-gemini-api-key"
streamlit run app.py
```

## Run in Google Colab (for quick testing only)

Colab + a tunnel (ngrok) is fine for trying the app out, but the link dies
the moment the Colab runtime stops — see "Deploy" below for a permanent
link.

```python
!pip install -r requirements.txt pyngrok

from pyngrok import ngrok
from getpass import getpass
import os

os.environ["GOOGLE_API_KEY"] = getpass("Gemini API key: ")
ngrok.set_auth_token(getpass("ngrok auth token: "))

get_ipython().system_raw("streamlit run app.py &>/content/logs.txt &")
print(ngrok.connect(8501))
```

## Deploy permanently — no Colab, no ngrok

1. Push this whole folder to GitHub (`.gitignore` already keeps your API
   key and the local `chroma_db/` out of the repo — `assets/` **should**
   be committed, it's just the logo).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** →
   pick the repo → main file `app.py`.
3. In **Settings → Secrets**, add:
   ```toml
   GOOGLE_API_KEY = "your-gemini-api-key"
   ```
4. Deploy. You get a permanent `https://your-app.streamlit.app` link that
   stays up on its own.

## How uploaded documents are processed

When you switch to **📂 My Uploaded Document** and upload a PDF, TXT, or
DOCX and click **Analyze Document**:

1. Text is extracted (`PyPDFLoader` for PDF, `python-docx` for DOCX, plain
   read for TXT).
2. It's cleaned and split into ~1000-character chunks with 150-character
   overlap — the same settings as the Psychology 2e book.
3. Chunks are embedded with the same `all-MiniLM-L6-v2` model and stored
   in a fresh, in-memory Chroma collection unique to that upload.
4. Every question you ask in this mode retrieves only from that
   collection — it never mixes with the Psychology 2e database.

Switching back to "Psychology Book" (or uploading a new file) simply
points the app at a different vector store; nothing is deleted.

## Security note

Never commit real API keys or tokens to a public GitHub repo. `app.py`
reads `GOOGLE_API_KEY` from Streamlit secrets or an environment variable —
it is never hardcoded. If an earlier version of this project had a key
typed directly into the code, regenerate that key, since anyone who saw it
could use it under your account.
