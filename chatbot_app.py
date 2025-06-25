import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from utils.loaders import load_url_content
from utils.llm_chain import get_combined_response
from load_vectorstore_from_gcp import load_vectorstore_from_gcp
import tempfile
from logging import Logger

import streamlit as st
import uuid
import time

# Real-time user tracking
if 'active_users' not in st.session_state:
    st.session_state.active_users = {}

if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

def update_active_users():
    now = time.time()
    ttl = 60
    active_users = {
        uid: ts for uid, ts in st.session_state.active_users.items()
        if now - ts < ttl
    }
    active_users[st.session_state.user_id] = now
    st.session_state.active_users = active_users
    return len(active_users)

active_user_count = update_active_users()

# Show at top-right
st.markdown(
    f"""
    <div style="position:fixed; top:10px; right:20px; background:#f0f0f0; padding:6px 12px; border-radius:8px; font-size:14px;">
        👥 Active Users: {active_user_count}
    </div>
    """, unsafe_allow_html=True
)



# After fetching GCP config
logger = Logger(
    gcp_bucket=gcp_config["bucket_name"],
    gcp_creds=gcp_config["credentials"],
    base_path="logs"
)


# ─────────────────────────────────────────────
# Helper: preview docs inside the TimesPro vectorstore
# ─────────────────────────────────────────────
def _preview_vectorstore(retriever, n_docs=5, char_limit=5000):
    try:
        docs = list(retriever.vectorstore.docstore._dict.values())[:n_docs]
        joined = "\n\n--- DOC SPLIT ---\n\n".join(d.page_content for d in docs)
        return joined[:char_limit]
    except Exception:
        return "⚠️ Could not preview vectorstore content."

# ====== SECRETS & GCP CREDENTIALS ======
openai_key = st.secrets["OPENAI_API_KEY"]
gcp_credentials_dict = dict(st.secrets["GCP_SERVICE_ACCOUNT"])

gcp_config = {
    "bucket_name": "test_bucket_brian",
    "prefix": "vectorstores",
    "credentials": gcp_credentials_dict,
}

# ====== PAGE CONFIG ======
st.set_page_config(page_title="AI Sales Assistant", layout="centered")
st.title("📚 AI Sales Assistant")

# ====== LLM MODEL (fixed to gpt‑4o) ======
model_choice = "gpt-4o"  # default model
# (model dropdown removed)

# ====== INPUTS ======
# — PDF upload removed —
# pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

timespro_urls = [
    "-- Select a program --",
    "https://timespro.com/executive-education/iim-calcutta-senior-management-programme",
    "https://timespro.com/executive-education/iim-kashipur-senior-management-programme",
    "https://timespro.com/executive-education/iim-raipur-senior-management-programme",
    "https://timespro.com/executive-education/iim-indore-senior-management-programme",
    "https://timespro.com/executive-education/iim-kozhikode-strategic-management-programme-for-cxos",
    "https://timespro.com/executive-education/iim-calcutta-lead-an-advanced-management-programme",
]

# Display only slug after /executive-education/
def slug(url):
    return url.split("/executive-education/")[1] if "/executive-education/" in url else url

display_options = [("-- Select a program --")] + [slug(u) for u in timespro_urls[1:]]
sel_display = st.selectbox("Select TimesPro Program", display_options, index=0)
url_1 = None if sel_display == "-- Select a program --" else timespro_urls[display_options.index(sel_display)]
url_2 = st.text_input("Input Competitor Program URL")

# ====== SANITIZE URL ======
def sanitize_url(url: str) -> str:
    return url.strip("/").split("/")[-1].replace("-", "_")

# ====== LOAD VECTORSTORE ======
retriever = None
if url_1:
    folder = f"timespro_com_executive_education_{sanitize_url(url_1)}"
    with st.spinner("Loading TimesPro vectorstore …"):
        try:
            vect_path = f"{gcp_config['prefix']}/{folder}"
            vectorstore = load_vectorstore_from_gcp(
                bucket_name=gcp_config["bucket_name"],
                path=vect_path,
                creds_dict=gcp_config["credentials"],
            )
            retriever = vectorstore.as_retriever(search_kwargs={"k": 50, "fetch_k": 100})
            st.success("Vectorstore loaded ✔️")
        except Exception as e:
            st.error(f"Vectorstore load failed: {e}")

# ====== SESSION STATE ======
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", input_key="question", output_key="answer",
        return_messages=True, k=7
    )
st.session_state.setdefault("comparison_output", "")
st.session_state.setdefault("comparison_injected", False)

# ====== ACTION BUTTONS ======
col_cmp, col_prn, col_clr = st.columns([2, 2, 1])
with col_cmp:
    cmp_clicked = st.button("Compare (Generate Brief)", disabled=sel_display == "-- Select a program --")
with col_prn:
    prn_clicked = st.button("Print Extracted Data")
with col_clr:
    if st.button("Clear Cache 🧹"):
        st.session_state.clear(); st.rerun()

# ====== PRINT EXTRACTED DATA ======
if prn_clicked:
    st.subheader("📄 TimesPro Data Preview")
    st.write(_preview_vectorstore(retriever) if retriever else "No vectorstore loaded.")
    st.subheader("📄 Competitor Data Preview")
    comp_txt = load_url_content([url_2]).get(url_2, "No competitor data.") if url_2 else "No competitor URL."
    st.write(comp_txt[:5000])

# ====== COMPARISON LOGIC ======
if cmp_clicked:
    if not url_2:
        st.warning("Please enter a competitor program URL.")
    else:
        with st.spinner("Generating sales‑enablement brief …"):
            pdf_text = ""  # PDF feature disabled
            url_texts = load_url_content([url_1, url_2])
            st.session_state.comparison_output = get_combined_response(
                pdf_text, url_texts, timespro_url=url_1, competitor_url=url_2, model_choice=model_choice
            )
            logger.log_metadata(url_1, url_2)
            logger.log_comparison_output(st.session_state.comparison_output)
            st.session_state.comparison_injected = False

# ====== DISPLAY BRIEF ======
if st.session_state.comparison_output:
    st.success("### 📝 Sales‑Enablement Brief")
    st.write(st.session_state.comparison_output)

# ====== CHATBOT SECTION ======
st.subheader("💬 Ask a follow‑up question")
user_q = st.text_input("Enter your question")

if user_q:
    if not retriever:
        st.warning("Vectorstore unavailable. Select a TimesPro program first.")
    else:
        with st.spinner("Answering …"):
            tp_ctx = load_url_content([url_1]).get(url_1, "")
            comp_ctx = load_url_content([url_2]).get(url_2, "")
            comparison_ctx = st.session_state.comparison_output

            # Lightweight open-ended prompt
            system_prompt = f"""
You are a smart, sales-savvy AI assistant helping learners and internal sales teams understand and compare educational programs.

You're informed by three sources:
1. Vectorstore-based TimesPro documents (for factual answers).
2. Web content from TimesPro and competitor URLs (for additional insights).
3. A previously generated sales brief (optional).

Use all relevant info to provide insightful, strategic, and helpful answers. Intelligently fill in with general knowledge even when the details aren't available try to tell something related to it from the document, rather than saying no infromation available. Keep the tone clear, concise, and helpful.

Avoid mentioning platform names like Coursera, Emeritus, etc.

--- TIMESPRO VECTORSTORE ---
(Used internally in chain)

--- TIMESPRO URL CONTENT ---
{tp_ctx}

--- COMPETITOR URL CONTENT ---
{comp_ctx}

--- SALES BRIEF (if any) ---
{comparison_ctx}
"""

            llm = ChatOpenAI(model_name=model_choice, openai_api_key=openai_key, temperature=0.4)

            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=st.session_state.memory,
                return_source_documents=True
            )

            # Inject the system prompt only once
            if not st.session_state.comparison_injected:
                st.session_state.memory.chat_memory.add_user_message("SYSTEM CONTEXT")
                st.session_state.memory.chat_memory.add_ai_message(system_prompt)
                st.session_state.comparison_injected = True

            # Call the chain
            answer = qa_chain.invoke({"question": user_q})
            st.write(f"💬 **Answer:** {answer['answer']}")

            if "qa_pairs" not in st.session_state:
                st.session_state.qa_pairs = []

st.session_state.qa_pairs.append((user_q, answer['answer']))

if st.session_state.get("qa_pairs") and st.session_state.get("comparison_output"):
    logger.log_chatbot_qa(st.session_state.qa_pairs)
    gcs_log_path = logger.write_to_gcs()
    st.info(f"📝 Log saved to GCS: `{gcs_log_path}`")
