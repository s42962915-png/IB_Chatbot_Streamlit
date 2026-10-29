import streamlit as st
from groq import Groq

from rag import KnowledgeBase

st.set_page_config(
    page_title="InBody Knowledge Assistant",
    page_icon="\U0001F4D8",
    layout="centered",
)

# --- InBody branding & theme -------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Lato:wght@300;400;700;900&display=swap');
    html, body, [class*="css"], .stApp { font-family: 'Lato', 'Source Sans Pro', sans-serif !important; }
    .stApp { background-color: #F7FAFE; }
    [data-testid="stHeader"] { background: linear-gradient(90deg, #0057B8 0%, #003D82 100%); }
    .stButton > button { background-color: #0057B8; color: #ffffff; border: none; border-radius: 8px; font-weight: 700; }
    .stButton > button:hover { background-color: #003D82; color: #ffffff; border: none; }
    [data-testid="stSidebar"] { background-color: #EEF4FC; }
    [data-testid="stChatMessage"] { border-radius: 14px; }
    h1, h2, h3 { color: #003D82 !important; }
    #inbody-footer { margin-top: 28px; padding-top: 12px; border-top: 2px solid #0057B8; text-align: center; color: #5b6b7c; font-size: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="background:linear-gradient(90deg,#0057B8 0%,#003D82 100%);color:#ffffff;padding:18px 24px;border-radius:14px;margin-bottom:14px;">
      <div style="font-size:12px;letter-spacing:2px;font-weight:700;opacity:.9;">INBODY</div>
      <div style="font-size:26px;font-weight:900;line-height:1.2;">Knowledge Assistant</div>
      <div style="font-size:13px;opacity:.85;margin-top:4px;">Answers grounded in the approved InBody documents &middot; no web search</div>
    </div>
    """,
    unsafe_allow_html=True,
)

SYSTEM_PROMPT = (
    "You are the InBody Knowledge Assistant.\n"
    "Answer ONLY from the document excerpts provided in the CONTEXT section.\n"
    "- Use only information explicitly present in the CONTEXT.\n"
    "- Do not use your own general knowledge and do not browse the web.\n"
    '- If the CONTEXT does not support an answer, reply exactly: "I couldn\'t find '
    'enough information about this in the approved InBody documents. Please check '
    'with the relevant InBody team."\n'
    "- If sources conflict, say so and identify the conflicting files; do not guess.\n"
    "- Cite the source filename(s) you used.\n"
    "- Keep answers concise. Use a table for comparisons.\n"
    "- Do not invent specifications, page numbers, or slide numbers."
)

REFUSAL = (
    "I couldn't find enough information about this in the approved InBody documents. "
    "Please check with the relevant InBody team."
)


def get_secret(name, default=None):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


@st.cache_resource(show_spinner=False)
def load_kb(kb_dir: str):
    return KnowledgeBase(kb_dir)


def build_prompt(question: str, passages) -> str:
    if not passages:
        return "QUESTION:\n" + question + "\n\nNo matching excerpts were retrieved."
    blocks = []
    for filename, chunk in passages:
        blocks.append("[source: " + filename + "]\n" + chunk)
    context = "\n\n".join(blocks)
    return "CONTEXT (excerpts from approved InBody documents):\n" + context + "\n\nQUESTION:\n" + question


# --- configuration ---------------------------------------------------------
api_key = get_secret("GROQ_API_KEY")
model = get_secret("GROQ_MODEL", "qwen/qwen3.8-27b")
kb_dir = get_secret("KB_DIR", "knowledge_base")
top_k = int(get_secret("TOP_K", "6"))

# Header is rendered by the InBody banner above.

kb = load_kb(kb_dir)
if kb.chunk_count == 0:
    st.warning("No documents found in the knowledge base folder yet.")
else:
    st.caption(
        str(kb.document_count)
        + " document(s) indexed - "
        + str(kb.chunk_count)
        + " passages ready"
    )

if not api_key:
    st.error("Add your Groq API key to Streamlit Secrets (GROQ_API_KEY) to start answering.")
    st.code('GROQ_API_KEY = "gsk_..."', language="toml")
    st.stop()

client = Groq(api_key=api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Knowledge base")
    st.write("Answers come only from the documents in the knowledge base folder.")
    st.caption("No web search is used.")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("evidence"):
            with st.expander("Sources"):
                for e in msg["evidence"]:
                    st.markdown("**" + e["filename"] + "**")
                    st.caption(e["snippet"])

question = st.chat_input("Ask about an InBody model, feature, or specification...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the approved InBody documents..."):
            passages = kb.search(question, top_k=top_k)
            prompt = build_prompt(question, passages)
            evidence = [
                {"filename": p[0], "snippet": p[1][:260]}
                for p in passages
            ]
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=1024,
                )
                answer = (response.choices[0].message.content or "").strip() or REFUSAL
            except Exception as exc:
                answer = "The request failed. Check your Groq API key and model name."
                st.caption("Technical detail: " + str(exc))

            st.markdown(answer)
            if evidence:
                with st.expander("Sources"):
                    for e in evidence:
                        st.markdown("**" + e["filename"] + "**")
                        st.caption(e["snippet"])
            else:
                st.caption("No matching passages found in the knowledge base.")

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "evidence": evidence}
            )

st.markdown(
    """
    <div id="inbody-footer">Created by <strong style="color:#003D82;">Sakshi Mhatre</strong>, Sr Dietitian &mdash; for internal usage only.</div>
    """,
    unsafe_allow_html=True,
)
