import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from utils.loaders import load_url_content
from utils.llm_chain import get_combined_response
from load_vectorstore_from_gcp import load_vectorstore_from_gcp
from custom_logger import Logger
import uuid
import time
from datetime import datetime
import platform

if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()

# === Real-time user tracking ===
if 'active_users' not in st.session_state:
    st.session_state.active_users = {}

if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

if "qa_pairs" not in st.session_state:
    st.session_state.qa_pairs = []

if "comparison" not in st.session_state:
    st.session_state.comparison = None

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

st.markdown(
    f"""
    <div style="position:fixed; top:10px; left:20px; background:#031830; padding:6px 12px; border-radius:8px; font-size:14px; z-index:9999;">
        👥 Active Users: {active_user_count}
    </div>
    """, unsafe_allow_html=True
)

# === Helper: preview vectorstore ===
def _preview_vectorstore(retriever, n_docs=5, char_limit=5000):
    try:
        docs = list(retriever.vectorstore.docstore._dict.values())[:n_docs]
        joined = "\n\n--- DOC SPLIT ---\n\n".join(d.page_content for d in docs)
        return joined[:char_limit]
    except Exception:
        return "⚠️ Could not preview vectorstore content."

# === Secrets & Logger ===
openai_key = st.secrets["OPENAI_API_KEY"]
gcp_credentials_dict = dict(st.secrets["GCP_SERVICE_ACCOUNT"])

gcp_config = {
    "bucket_name": "test_bucket_brian",
    "prefix": "vectorstores",
    "credentials": gcp_credentials_dict,
}

logger = Logger(
    gcp_bucket=gcp_config["bucket_name"],
    gcp_creds=gcp_config["credentials"],
    base_path="logs"
)

# === Page config ===
st.set_page_config(page_title="AI Sales Assistant", layout="centered")
st.title("📚 AI Sales Assistant")

# === Inputs ===
timespro_urls = [
    "-- Select a program --",
    "https://timespro.com/executive-education/iim-calcutta-senior-management-programme",
    "https://timespro.com/executive-education/iim-kashipur-senior-management-programme",
    "https://timespro.com/executive-education/iim-raipur-senior-management-programme",
    "https://timespro.com/executive-education/iim-indore-senior-management-programme",
    "https://timespro.com/executive-education/iim-kozhikode-strategic-management-programme-for-cxos",
    "https://timespro.com/executive-education/iim-calcutta-lead-an-advanced-management-programme",
]

def slug(url):
    return url.split("/executive-education/")[1] if "/executive-education/" in url else url

display_options = ["-- Select a program --"] + [slug(u) for u in timespro_urls[1:]]
sel_display = st.selectbox("Select TimesPro Program", display_options, index=0)
url_1 = None if sel_display == "-- Select a program --" else timespro_urls[display_options.index(sel_display)]
url_2 = st.text_input("Input Competitor Program URL")

def sanitize_url(url: str) -> str:
    return url.strip("/").split("/")[-1].replace("-", "_")

# === Load vectorstore ===
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

# === Session state memory ===
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", input_key="question", output_key="answer",
        return_messages=True, k=7
    )

st.session_state.setdefault("comparison_output", "")
st.session_state.setdefault("comparison_injected", False)

# === Action Buttons ===
col_cmp, col_prn, col_clr = st.columns([2, 2, 1])
with col_cmp:
    cmp_clicked = st.button("Compare (Generate Brief)", disabled=sel_display == "-- Select a program --")
with col_prn:
    prn_clicked = st.button("Print Extracted Data")
with col_clr:
    if st.button("Clear Cache 🪩"):
        st.session_state.clear(); st.rerun()

# === Print extracted data ===
if prn_clicked:
    st.subheader("📄 TimesPro Data Preview")
    st.write(_preview_vectorstore(retriever) if retriever else "No vectorstore loaded.")
    st.subheader("📄 Competitor Data Preview")
    comp_txt = load_url_content([url_2]).get(url_2, "No competitor data.") if url_2 else "No competitor URL."
    st.write(comp_txt[:5000])

# === Comparison logic ===
if cmp_clicked:
    if not url_2:
        st.warning("Please enter a competitor program URL.")
    else:
        with st.spinner("Generating sales‑enablement brief …"):
            pdf_text = ""
            url_texts = load_url_content([url_1, url_2])
            st.session_state.comparison_output = get_combined_response(
                pdf_text, url_texts, timespro_url=url_1, competitor_url=url_2, model_choice="gpt-4o"
            )
            logger.log_metadata(url_1, url_2)
            logger.log_comparison_output(st.session_state.comparison_output)
            st.session_state.comparison_injected = False

# === Display brief ===
if st.session_state.comparison_output:
    st.success("### 📝 Sales‑Enablement Brief")
    st.write(st.session_state.comparison_output)

# === Chatbot section ===
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

            system_prompt = f"""
You are a smart, sales-savvy AI assistant helping learners and internal sales teams understand and compare educational programs.
You're informed by three sources:
1. Vectorstore-based TimesPro documents (for factual answers).
2. Web content from TimesPro and competitor URLs (for additional insights).
3. A previously generated sales brief (optional).

Use all relevant info to provide insightful, strategic, and helpful answers. Intelligently fill in with general knowledge even when the details aren't available. Keep the tone clear, concise, and helpful.
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

            llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=openai_key, temperature=0.4)
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=st.session_state.memory,
                return_source_documents=True
            )

            if not st.session_state.comparison_injected:
                st.session_state.memory.chat_memory.add_user_message("SYSTEM CONTEXT")
                st.session_state.memory.chat_memory.add_ai_message(system_prompt)
                st.session_state.comparison_injected = True

            answer = qa_chain.invoke({"question": user_q})
            st.write(f"💬 **Answer:** {answer['answer']}")
            st.session_state.qa_pairs.append((user_q, answer['answer']))
            
            # Gather metadata
            session_id = st.session_state.user_id
            session_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            runtime = time.time() - st.session_state.get("start_time", time.time())
            device_info = platform.platform()
            
            metadata = {
                "session_id": session_id,
                "timestamp": session_time,
                "device": device_info,
                "session_runtime_seconds": round(runtime),
                "selected_program": url_1,
                "competitor_program": url_2,
                "comparison_output": st.session_state.comparison_output,
                "active_user_count": active_user_count,
                "qa_pairs": st.session_state.qa_pairs,
            }

            logger.log_chatbot_qa(st.session_state.qa_pairs)
            gcs_log_path = logger.write_to_gcs()
            st.session_state.log_saved = True
            st.info(f"📝 Log saved to GCS: `{gcs_log_path}`")

#if st.session_state.get("qa_pairs") and st.session_state.get("comparison_output"):
#   logger.log_chatbot_qa(st.session_state.qa_pairs)
#    gcs_log_path = logger.write_to_gcs()
#    st.info(f"📝 Log saved to GCS: `{gcs_log_path}`")
