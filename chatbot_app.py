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
    with st.spinner("Loading TimesPro program details..."):
        try:
            vectorstore_path = f"{gcp_config['prefix']}/{folder_name}"
            vectorstore = load_vectorstore_from_gcp(
                bucket_name=gcp_config["bucket_name"],
                path=vectorstore_path,
                creds_dict=gcp_config["credentials"]
            )
            retriever = vectorstore.as_retriever()
            st.success("Vectorstore loaded successfully.")
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

if "comparison_output" not in st.session_state:
    st.session_state.comparison_output = ""

if "comparison_injected" not in st.session_state:
    st.session_state.comparison_injected = False

# ====== COMPARISON & CLEAR CACHE BUTTONS SIDE BY SIDE ======
col_compare, col_clear = st.columns([3, 1])

with col_compare:
    compare_disabled = selected_program == "-- Select a program --"
    compare_clicked = st.button("Compare", disabled=compare_disabled)

with col_clear:
    clear_clicked = st.button("Clear Cache 🧹", help="This will reset chat history and comparison")

# ====== CLEAR CACHE CONFIRMATION ======
if clear_clicked:
    st.session_state.show_confirm_clear = True

if st.session_state.get("show_confirm_clear", False):
    with st.expander("⚠️ Confirm Clear Cache", expanded=True):
        st.warning("Are you sure you want to clear chat and comparison history?")
        col_confirm, col_cancel = st.columns(2)
        with col_confirm:
            if st.button("Yes, clear it", key="confirm_clear"):
                st.session_state.clear()
                st.rerun()
        with col_cancel:
            if st.button("Cancel", key="cancel_clear"):
                st.session_state.show_confirm_clear = False

# ====== HANDLE COMPARISON LOGIC ======
if compare_clicked:
    if selected_program == "-- Select a program --":
        st.warning("Please select a valid TimesPro program from the dropdown.")
    elif not (pdf_file or url_1 or url_2):
        st.warning("Please upload a PDF or enter at least one URL.")
    else:
        with st.spinner("Comparing TimesPro program with competitors..."):
            pdf_text = ""
            if pdf_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                    tmp_pdf.write(pdf_file.read())
                    pdf_path = tmp_pdf.name
                pdf_text = load_pdf(pdf_path)

            url_texts = load_url_content([url_1, url_2]) or {}
            response = get_combined_response(pdf_text, url_texts, model_choice=model_choice)
            st.session_state.comparison_output = response
            st.session_state.comparison_injected = False

# ====== DISPLAY COMPARISON OUTPUT IF EXISTS ======
if st.session_state.get("comparison_output"):
    st.success("Here's the comparison:")
    st.write(st.session_state.comparison_output)

# ====== QA CHATBOT SECTION ======
st.subheader("💬 Ask a follow-up question about the TimesPro program")
user_question = st.text_input("Enter your question here")

if user_question:
    if not retriever:
        st.warning("Knowledge base is not available. Please select a program to load its data.")
    else:
        with st.spinner("Answering your question using all available information..."):

            pdf_text = ""
            if pdf_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                    tmp_pdf.write(pdf_file.read())
                    pdf_path = tmp_pdf.name
                pdf_text = load_pdf(pdf_path)

            url_contexts = load_url_content([url_1, url_2]) or {}
            timespro_context = url_contexts.get(url_1, "")
            competitor_context = url_contexts.get(url_2, "")
            comparison_context = st.session_state.comparison_output or ""

            system_prompt = """You are an expert EdTech counselor. You must answer user queries based on the provided TimesPro course content, competitor details, and the PDF brochure if available.

Use the following context:

--- TIMESPRO COURSE CONTENT ---
{timespro_context}

--- COMPETITOR PROGRAM DETAILS ---
{competitor_context}

--- PDF BROCHURE CONTENT ---
{pdf_context}

--- COMPARISON OUTPUT ---
{comparison_context}

Only answer using the above context. Be accurate, neutral, and helpful.
If you don't know something, say so honestly.
"""

            formatted_prompt = system_prompt.format(
                timespro_context=timespro_context,
                competitor_context=competitor_context,
                pdf_context=pdf_text,
                comparison_context=comparison_context,
            )

            custom_llm = ChatOpenAI(
                model_name=model_choice,
                openai_api_key=openai_key,
                temperature=0
            )

            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=custom_llm,
                retriever=retriever,
                memory=st.session_state.memory,
                return_source_documents=True
            )

            if not st.session_state.comparison_injected:
                st.session_state.memory.chat_memory.add_user_message("System Prompt with Full Context")
                st.session_state.memory.chat_memory.add_ai_message(formatted_prompt)
                st.session_state.comparison_injected = True

            full_question = formatted_prompt + "\n\nUser Question: " + user_question
            result = qa_chain.invoke({"question": full_question})
            st.write(f"💬 **Answer:** {result['answer']}")
