import streamlit as st
from groq import Groq

from rag import KnowledgeBase

st.set_page_config(
    page_title="InBody Knowledge Assistant",
    page_icon="\U0001F4D8",
    layout="centered",
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

st.title("\U0001F4D8 InBody Knowledge Assistant")
st.caption("Answers are grounded in the approved InBody documents. No web search.")

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
