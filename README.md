# Psychology RAG 📚

A Retrieval-Augmented Generation (RAG) app that answers questions about the
OpenStax *Psychology 2e* textbook, with cited sources.

**Pipeline:** PDF → clean & chunk → embed (`all-MiniLM-L6-v2`) → Chroma vector
store → retrieve top-10 → rerank with a cross-encoder → top-4 → Gemini
(`gemini-3.5-flash`) generates a cited answer → Streamlit UI.

## Files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit app (UI + RAG pipeline) |
| `vectorstore_utils.py` | Downloads the PDF and builds/loads the Chroma vector store |
| `requirements.txt` | Python dependencies |
| `.streamlit/secrets.toml.example` | Shows which secret key the app needs |

The vector store is **not** stored in the repo — the app builds it
automatically on its first run (downloads the PDF, chunks it, embeds it),
then reuses it on every run after that.

## Run locally

```bash
pip install -r requirements.txt
export GOOGLE_API_KEY="your-gemini-api-key"
streamlit run app.py
```

## Deploy so it's always on — no Colab, no ngrok

Colab + ngrok only work while the notebook is open and re-run — the link
breaks the moment the session ends, which is why you had to reconnect
ngrok every time. To have a permanent link that just works:

1. Push this repo to GitHub (see `.gitignore` — your API key is never
   committed).
2. Go to [share.streamlit.io](https://share.streamlit.io) (Streamlit
   Community Cloud, free) → **New app** → pick this repo → main file
   `app.py`.
3. In the app's **Settings → Secrets**, paste:
   ```toml
   GOOGLE_API_KEY = "your-gemini-api-key"
   ```
4. Deploy. You get a permanent `https://your-app.streamlit.app` URL that
   stays up on its own — nothing to run locally or reconnect.

(Hugging Face Spaces with the Streamlit SDK is an equally good free
alternative if you prefer it.)

## Security note

The original notebook had the Gemini API key and an ngrok auth token
hardcoded directly in the code. Both were **revoked from this version** —
never commit real API keys/tokens to a public GitHub repo. `app.py` now
reads the key from Streamlit secrets or an environment variable instead.
If you shared the notebook with the old hardcoded keys anywhere before,
regenerate both the Gemini API key and the ngrok token, since anyone who
saw them could use them under your account.
