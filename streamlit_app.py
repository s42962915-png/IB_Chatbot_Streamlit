import streamlit as st
from groq import Groq

from rag import KnowledgeBase

st.set_page_config(
    page_title="InBody Assistant",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

SYSTEM_PROMPT = (
    "You are the InBody Assistant.\n"
    "Answer ONLY from the document excerpts provided in the CONTEXT section.\n"
    "- Use only information explicitly present in the CONTEXT.\n"
    "- Do not use your own general knowledge and do not browse the web.\n"
    "- If the CONTEXT does not support an answer, reply exactly: \"I couldn't find "
    "enough information about this in the approved InBody documents. Please check "
    "with the relevant InBody team.\"\n"
    "- If sources conflict, say so and identify the conflicting files; do not guess.\n"
    "- Cite the source filename(s) you used.\n"
    "- Keep answers concise. Use a table for comparisons.\n"
    "- Do not invent specifications, page numbers, or slide numbers."
)

REFUSAL = (
    "I couldn't find enough information about this in the approved InBody documents. "
    "Please check with the relevant InBody team."
)

THEME = """<style>
@import url('https://fonts.googleapis.com/css2?family=Lato:wght@300;400;700;900&display=swap');

#MainMenu, [data-testid="stHeader"], [data-testid="stToolbar"], footer { display: none !important; }

html, body, .stApp, [data-testid="stAppViewContainer"] {
    font-family: 'Lato', 'Source Sans Pro', sans-serif !important;
}
.stApp {
    background: radial-gradient(ellipse 120% 90% at 50% -10%, #43101f 0%, #23070d 45%, #0a0103 100%);
    color: #f3e7ea;
}
.block-container { padding-top: 1.0rem !important; max-width: 980px; }

.ib-topbar {
    display: flex; align-items: center; gap: 12px;
    padding: 10px 14px; border-radius: 12px;
    background: rgba(18,4,8,0.72);
    border: 1px solid rgba(120,40,55,0.35);
    margin-bottom: 26px;
}
.ib-logo {
    width: 32px; height: 32px; border-radius: 8px; flex: 0 0 auto;
    background: linear-gradient(135deg,#e0566f,#8d1f33);
    display: flex; align-items: center; justify-content: center;
    font-weight: 900; color: #fff; font-size: 14px;
}
.ib-brand-title { font-size: 15px; font-weight: 900; color: #ffffff; line-height: 1.15; }
.ib-brand-sub { font-size: 11.5px; color: #b3909a; }

.ib-hero { margin: 24px 0 6px 0; }
.ib-hero h1 { font-size: 44px; font-weight: 900; color: #ffffff; margin: 0 0 10px 0; letter-spacing: -1px; line-height: 1.05; }
.ib-hero p { font-size: 15px; color: #caa9b2; line-height: 1.55; max-width: 600px; }

.ib-cards-note { font-size: 11px; color: #8f6f78; margin: 14px 0 4px 0; }

div[data-testid="stButton"] > button {
    width: 100%;
    text-align: left !important;
    justify-content: flex-start !important;
    white-space: normal !important;
    background: rgba(30,8,14,0.85) !important;
    color: #ead5d9 !important;
    border: 1px solid rgba(120,40,55,0.32) !important;
    border-radius: 10px !important;
    padding: 12px 14px !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    line-height: 1.35 !important;
    min-height: 58px !important;
}
div[data-testid="stButton"] > button:hover {
    border-color: #c56b7d !important;
    background: rgba(60,14,24,0.9) !important;
    color: #ffffff !important;
}
div[data-testid="stButton"] > button[kind="secondary"] {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    color: #f3e7ea !important;
    min-height: unset !important;
    font-weight: 700 !important;
}

[data-testid="stChatInput"] textarea {
    background: rgba(20,4,9,0.9) !important;
    color: #f3e7ea !important;
    border: 1px solid rgba(120,40,55,0.45) !important;
    border-radius: 14px !important;
}
[data-testid="stChatMessage"] {
    border-radius: 12px;
    background: rgba(20,4,9,0.55);
    border: 1px solid rgba(90,25,40,0.4);
}

#inbody-footer {
    margin-top: 44px; padding-top: 14px;
    border-top: 1px solid rgba(120,40,55,0.35);
    text-align: center; color: #8f6f78; font-size: 11.5px;
}
</style>"""


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
    blocks = [("[source: " + fn + "]\n" + ch) for fn, ch in passages]
    return "CONTEXT (excerpts from approved InBody documents):\n" + "\n\n".join(blocks) + "\n\nQUESTION:\n" + question


api_key = get_secret("GROQ_API_KEY")
model = get_secret("GROQ_MODEL", "qwen/qwen3.8-27b")
kb_dir = get_secret("KB_DIR", "knowledge_base")
top_k = int(get_secret("TOP_K", "6"))

st.markdown(THEME, unsafe_allow_html=True)

kb = load_kb(kb_dir)

if "messages" not in st.session_state:
    st.session_state.messages = []

pending = st.session_state.pop("pending_question", None)

bar = st.columns([8, 2])
with bar[0]:
    st.markdown(
        '<div class="ib-topbar">'
        '<div class="ib-logo">IB</div>'
        '<div><div class="ib-brand-title">InBody Assistant</div>'
        '<div class="ib-brand-sub">Grounded in ' + str(kb.document_count) + ' InBody documents</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )
with bar[1]:
    if st.button("New chat", key="newchat"):
        st.session_state.messages = []
        st.rerun()

if not st.session_state.messages and not pending:
    st.markdown(
        '<div class="ib-hero">'
        '<h1>How can I help?</h1>'
        '<p>Ask about InBody devices, results sheets, parameters and procedures. '
        'Grounded in ' + str(kb.document_count) + ' InBody documents. '
        'Every answer cites the document it came from.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    cards = [
        ("COMPARE", "Compare InBody 270 and InBody 570"),
        ("PARAMETERS", "How is visceral fat area calculated?"),
        ("PROCEDURE", "What precautions apply when measuring ICU patients?"),
        ("PREPARATION", "How should a client prepare for a test?"),
    ]
    cols = st.columns(2, gap="small")
    for i, (cat, q) in enumerate(cards):
        with cols[i % 2]:
            if st.button(cat + "  -  " + q, key="card_" + str(i)):
                st.session_state.pending_question = q
                st.rerun()
    st.markdown(
        '<div class="ib-cards-note">Answers come only from InBody documents and can still be wrong. Check critical specs.</div>',
        unsafe_allow_html=True,
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("evidence"):
            with st.expander("Sources"):
                for e in msg["evidence"]:
                    st.markdown("**" + e["filename"] + "**")
                    st.caption(e["snippet"])

if not api_key:
    st.error("Add your Groq API key to Streamlit Secrets (GROQ_API_KEY) to start answering.")
    st.markdown('<div id="inbody-footer">Created by <strong style="color:#c56b7d;">Sakshi Mhatre</strong>, Sr Dietitian - for internal usage only.</div>', unsafe_allow_html=True)
    st.stop()

client = Groq(api_key=api_key)

typed = st.chat_input("Ask anything about InBody...")
prompt = typed or pending

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching the approved InBody documents..."):
            passages = kb.search(prompt, top_k=top_k)
            p_text = build_prompt(prompt, passages)
            evidence = [{"filename": p[0], "snippet": p[1][:260]} for p in passages]
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": p_text},
                    ],
                    temperature=0.1,
                    max_tokens=1024,
                )
                answer = (resp.choices[0].message.content or "").strip() or REFUSAL
            except Exception as exc:
                answer = "The request failed. Check your Groq API key and model name."
                st.caption("Technical detail: " + str(exc))
            st.markdown(answer)
            if evidence:
                with st.expander("Sources"):
                    for e in evidence:
                        st.markdown("**" + e["filename"] + "**")
                        st.caption(e["snippet"])
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "evidence": evidence}
            )

st.markdown(
    '<div id="inbody-footer">Created by <strong style="color:#c56b7d;">Sakshi Mhatre</strong>, Sr Dietitian - for internal usage only.</div>',
    unsafe_allow_html=True,
)
