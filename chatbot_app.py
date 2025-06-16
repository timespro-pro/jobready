import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from utils.loaders import load_pdf, load_url_content
from utils.llm_chain import get_combined_response
from load_vectorstore_from_gcp import load_vectorstore_from_gcp
import tempfile

# ====== SECRETS & GCP CREDENTIALS ======
openai_key = st.secrets["OPENAI_API_KEY"]
gcp_credentials_dict = dict(st.secrets["GCP_SERVICE_ACCOUNT"])

gcp_config = {
    "bucket_name": "test_bucket_brian",
    "prefix": "vectorstores",
    "credentials": gcp_credentials_dict
}

# ====== PAGE CONFIG ======
st.set_page_config(page_title="AI Sales Assistant", layout="centered")
st.title("📚 AI Sales Assistant")

# ====== MODEL CHOICE ======
model_choice = st.selectbox(
    "Choose a model",
    ["gpt-3.5-turbo", "gpt-4", "gpt-4-1106-preview", "gpt-4o"],
    index=0
)

# ====== INPUTS ======
pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

program_options = [
    "-- Select a program --",
    "https://timespro.com/executive-education/iim-calcutta-senior-management-programme",
    "https://timespro.com/executive-education/iim-kashipur-senior-management-programme",
    "https://timespro.com/executive-education/iim-raipur-senior-management-programme",
    "https://timespro.com/executive-education/iim-indore-senior-management-programme",
    "https://timespro.com/executive-education/iim-kozhikode-strategic-management-programme-for-cxos",
    "https://timespro.com/executive-education/iim-calcutta-lead-an-advanced-management-programme",
]

selected_program = st.selectbox("Select TimesPro Program URL", program_options, index=0)
url_1 = selected_program if selected_program != "-- Select a program --" else None
url_2 = st.text_input("Input Competitors Program URL")

# ====== SANITIZE URL ======
def sanitize_url(url: str) -> str:
    return url.strip("/").split("/")[-1].replace("-", "_")

# ====== LOAD VECTORSTORE ======
retriever = None
if url_1:
    folder_name = f"timespro_com_executive_education_{sanitize_url(url_1)}"
    with st.spinner("Loading TimesPro program details from Vectorstore ..."):
        try:
            vectorstore_path = f"{gcp_config['prefix']}/{folder_name}"
            vectorstore = load_vectorstore_from_gcp(
                bucket_name=gcp_config["bucket_name"],
                path=vectorstore_path,
                creds_dict=gcp_config["credentials"]
            )
            retriever = vectorstore.as_retriever()
            st.success("Vectorstore loaded successfully ✔️")
        except Exception as e:
            st.error(f"Vectorstore loading failed: {e}")

# ====== SESSION STATE INIT ======
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True,
        k=7
    )

state_defaults = {
    "comparison_output": "",
    "comparison_injected": False,
    "timespro_context": "",
    "competitor_context": "",
    "pdf_text": ""
}
for k, v in state_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ====== COMPARISON & CLEAR CACHE BUTTONS ======
col_compare, col_clear = st.columns([3, 1])

with col_compare:
    compare_disabled = selected_program == "-- Select a program --"
    compare_clicked = st.button("Compare", disabled=compare_disabled)

with col_clear:
    clear_clicked = st.button("Clear Cache 🧹", help="This will reset chat, comparison, and cached contexts")

if clear_clicked:
    st.session_state.clear()
    st.rerun()

# ====== HELPERS TO CACHE CONTEXTS LOCALLY (ONE‑OFF) ======

def ensure_timespro_context():
    if url_1 and not st.session_state.timespro_context:
        st.session_state.timespro_context = load_url_content([url_1]).get(url_1, "")

def ensure_competitor_context():
    if url_2 and not st.session_state.competitor_context:
        st.session_state.competitor_context = load_url_content([url_2]).get(url_2, "")

def ensure_pdf_text():
    if pdf_file and not st.session_state.pdf_text:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            tmp_pdf.write(pdf_file.read())
            pdf_path = tmp_pdf.name
        pdf_text = load_pdf(pdf_path)
        st.session_state.pdf_text = pdf_text if pdf_text.strip() else "No content extracted from the uploaded PDF."

# ====== COMPARISON LOGIC ======
if compare_clicked:
    if not (pdf_file or url_1 or url_2):
        st.warning("Please upload a PDF or enter at least one URL.")
    else:
        with st.spinner("Comparing TimesPro program with competitor ..."):
            # Ensure all contexts are loaded once
            ensure_pdf_text()
            ensure_timespro_context()
            ensure_competitor_context()

            response = get_combined_response(
                st.session_state.pdf_text,
                {url_1: st.session_state.timespro_context, url_2: st.session_state.competitor_context},
                model_choice=model_choice,
            )
            st.session_state.comparison_output = response
            st.session_state.comparison_injected = False  # reset prompt injection

# ====== DISPLAY COMPARISON ======
if st.session_state.comparison_output:
    st.success("### 📝 Comparison Result")
    st.write(st.session_state.comparison_output)

# ====== CHATBOT SECTION ======
st.subheader("💬 Chat with AI Sales Assistant")
user_prompt = st.text_input("Enter your message")

if user_prompt:
    if not retriever:
        st.warning("Knowledge base is not available. Please select a program to load its data.")
    else:
        with st.spinner("Processing your request ..."):
            # Make sure contexts are cached
            ensure_pdf_text()
            ensure_timespro_context()
            ensure_competitor_context()

            comparison_context = st.session_state.comparison_output or "No comparison data available."

            system_prompt = f"""
You are an expert EdTech counselor.
Answer user queries using ONLY the provided context. Be accurate, neutral, and helpful. If you don't know, say so honestly.

--- TIMESPRO DATA ---
{st.session_state.timespro_context or 'No data extracted from TimesPro URL.'}

--- COMPETITOR DATA ---
{st.session_state.competitor_context or 'No data extracted from competitor URL.'}

--- PDF DATA ---
{st.session_state.pdf_text or 'No content extracted from the uploaded PDF.'}

--- COMPARISON RESULT ---
{comparison_context}
"""

            # Initialise LLM & QA Chain
            custom_llm = ChatOpenAI(
                model_name=model_choice,
                openai_api_key=openai_key,
                temperature=0
            )

            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=custom_llm,
                retriever=retriever,
                memory=st.session_state.memory,
                return_source_documents=True,
            )

            # Inject system prompt once at start of conversation
            if not st.session_state.comparison_injected:
                st.session_state.memory.chat_memory.add_user_message("SYSTEM")
                st.session_state.memory.chat_memory.add_ai_message(system_prompt)
                st.session_state.comparison_injected = True

            # DEBUG SIZES
            st.write(f"TIMESPRO CONTEXT LENGTH: {len(st.session_state.timespro_context)}")
            st.write(f"COMPETITOR CONTEXT LENGTH: {len(st.session_state.competitor_context)}")
            st.write(f"PDF TEXT LENGTH: {len(st.session_state.pdf_text)}")
            st.write(f"COMPARISON LENGTH: {len(comparison_context)}")

            result = qa_chain.invoke({"question": user_prompt})
            st.write(f"💬 **Answer:** {result['answer']}")
