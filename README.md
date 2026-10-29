# IB_Chatbot - InBody Knowledge Assistant (Streamlit)

A **standalone Streamlit** chatbot that answers questions **only** from the approved InBody documents, using the **Groq free-tier** LLM for generation and **local (free) retrieval** for search.

- **Grounded, not web** - the model only ever sees excerpts retrieved from your documents. No web search, no browsing.
- **Genuinely free** - Groq free tier (no credit card) + fully local BM25 retrieval. No OpenAI, no per-request cost.
- **Low hallucination** - answers are constrained to the retrieved text and the assistant refuses when the documents don't support an answer.
- **Handles a few hundred requests/day** - well within Groq's free-tier limits.

---

## Quick start (under 15 minutes)

### 0. Prerequisites
- Python 3.10+
- A free Groq API key from https://console.groq.com/keys
- 2-5 approved InBody documents (PDF, PPTX, DOCX, TXT, or MD)

### 1. Clone & install

```bash
git clone <this-repo>
cd <this-repo>
python -m venv .venv
```

Activate it:
- Windows: `.venv\Scripts\activate`
- macOS/Linux: `source .venv/bin/activate`

```bash
pip install -r requirements.txt
```

### 2. Add your documents

Drop approved InBody files into `knowledge_base/`. The app parses them locally (no upload to any provider for indexing).

### 3. Configure the Groq key

Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "gsk_your_key_here"
GROQ_MODEL = "qwen/qwen3.8-27b"
```

### 4. Run

```bash
streamlit run streamlit_app.py
```

Open the printed local URL (usually http://localhost:8501).

---

## How it works

```text
Question
   |
   v
1. Retrieve: BM25 ranks passages from knowledge_base/ (local, free)
   v
2. Prompt: top passages + question are sent to Groq (no web)
   v
3. Answer: grounded reply + source filenames; refuses if unsupported
```

---

## Deploy to Streamlit Community Cloud (free)

1. Push this project to a GitHub repository.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. **New app** -> select the repository -> set the main file to `streamlit_app.py`.
4. In **Advanced settings -> Secrets**, paste:

```toml
GROQ_API_KEY = "gsk_your_key_here"
GROQ_MODEL = "qwen/qwen3.8-27b"
```

5. **Deploy**. You get a public URL like `https://your-app.streamlit.app`.

> Note: Streamlit Cloud is for a **standalone chat page**, not an embeddable widget. If you later need to embed on other websites, use the FastAPI + `widget.js` version instead.

---

## Configuration

| Secret | Default | Description |
| --- | --- | --- |
| `GROQ_API_KEY` | - | **Required.** Free key from Groq. |
| `GROQ_MODEL` | `qwen/qwen3.8-27b` | Groq model. Alternatives: `openai/gpt-oss-20b`, `openai/gpt-oss-120b`. |
| `KB_DIR` | `knowledge_base` | Folder with your documents. |
| `TOP_K` | `6` | Number of passages to retrieve per question. |

---

## Important caveats (read me)

- **No model is 100% hallucination-free.** This app minimizes it by (a) retrieving from your documents, (b) instructing the model to answer only from the retrieved text, and (c) refusing when unsupported. Always spot-check answers against the source.
- **Free-tier data usage.** Groq's free tier may use your prompts for service improvement. For confidential InBody material, get company approval, or run a local model (e.g. Ollama) instead of Groq.
- **Scanned PDFs.** Image-only PDFs have no text layer; run OCR on them first.
- **A few hundred requests/day** is fine, but Groq free tier has rate limits (roughly 30 req/min). Add a small queue/retry if you expect bursts.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| "No documents found" | Add PDF/PPTX/DOCX/TXT/MD files to `knowledge_base/`. |
| Answers are vague or wrong | Check the question's keywords; ensure the source docs contain the info. Increase `TOP_K`. |
| Request failed | Verify `GROQ_API_KEY` and model name; check Groq rate limits. |
| Scanned PDF returns nothing | OCR the PDF before adding it. |

---

## Project structure

```text
.
├── streamlit_app.py     # UI + Groq grounded generation
├── rag.py               # Document parsing + BM25 retrieval (local, free)
├── requirements.txt
├── .streamlit/secrets.toml.example
└── knowledge_base/      # Put approved documents here
```
