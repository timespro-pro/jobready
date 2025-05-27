import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from utils.loaders import load_pdf, load_url_content
from utils.llm_chain import get_combined_response
from load_vectorstore_from_gcp import load_vectorstore
import tempfile
import os

# ====== SECRETS ======
openai_key = st.secrets["OPENAI_API_KEY"]
gcp_config = {
    "bucket_name": "test_bucket_brian",
    "prefix": "vectorstores",
    "credentials": st.secrets["GCP_SERVICE_ACCOUNT"]
}
# =====================

# ====== PAGE CONFIG ======
st.set_page_config(page_title="AI Sales Assistant", layout="centered")
st.title("📚 AI Sales Assistant")
# =========================

# ====== MODEL CHOICE ======
model_choice = st.selectbox(
    "Choose a model",
    ["gpt-3.5-turbo", "gpt-4", "gpt-4-1106-preview", "gpt-4o"],
    index=0
)
# ==========================

# ====== INPUTS ======
pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

program_options = [
    "https://timespro.com/executive-education/iim-calcutta-senior-management-programme",
    "https://timespro.com/executive-education/iim-kashipur-senior-management-programme",
    "https://timespro.com/executive-education/iim-raipur-senior-management-programme",
    "https://timespro.com/executive-education/iim-indore-senior-management-programme",
    "https://timespro.com/executive-education/iim-kozhikode-strategic-management-programme-for-cxos",
    "https://timespro.com/executive-education/iim-calcutta-lead-an-advanced-management-programme",
]

selected_program = st.selectbox("Select TimesPro Program URL", program_options)
url_1 = selected_program
url_2 = st.text_input("Input URL 2 (Optional)")
# =====================

# ====== MAP TO GCP-FRIENDLY FOLDER NAME ======
def sanitize_url(url: str) -> str:
    return url.strip("/").split("/")[-1].replace("-", "_")

folder_name = f"timespro_com_executive_education_{sanitize_url(selected_program)}"
# =============================================

# ====== LOAD VECTORSTORE ======
with st.spinner("Loading TimesPro program details..."):
    try:
        vectorstore = load_vectorstore(folder_name=folder_name, openai_api_key=openai_key, gcp_config=gcp_config)
        retriever = vectorstore.as_retriever()
        st.success("Vectorstore loaded successfully.")
    except Exception as e:
        st.error(f"Vectorstore loading failed: {e}")
        retriever = None
# ==============================

# ====== INITIALIZE MEMORY (FIXED) ======
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=False,  # ✅ FIXED: changed to False to avoid ValueError
    k=7  # retain last 7 interactions
)
# ===============================

# ====== SESSION STATE ======
if "comparison_output" not in st.session_state:
    st.session_state.comparison_output = ""

if "comparison_injected" not in st.session_state:
    st.session_state.comparison_injected = False
# ===========================

# ====== MAIN COMPARISON ======
if st.button("Compare"):
    if not (pdf_file or url_1 or url_2):
        st.warning("Please upload a PDF or enter at least one URL.")
    else:
        with st.spinner("Comparing TimesPro program with competitors..."):
            pdf_text = ""
            if pdf_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                    tmp_pdf.write(pdf_file.read())
                    pdf_path = tmp_pdf.name
                pdf_text = load_pdf(pdf_path)

            url_texts = load_url_content([url_1, url_2])
            response = get_combined_response(pdf_text, url_texts, model_choice=model_choice)
            st.session_state.comparison_output = response
            st.session_state.comparison_injected = False  # reset so it can be re-injected
            st.success("Here's the comparison:")
            st.write(response)
# ====================================================

# ====== QA SECTION ======
st.subheader("💬 Ask a follow-up question about the TimesPro program")

user_question = st.text_input("Enter your question here")

if user_question and retriever:
    with st.spinner("Answering your question using the knowledge base..."):
        # Inject the comparison output into memory once
        if st.session_state.comparison_output and not st.session_state.comparison_injected:
            memory.chat_memory.add_user_message("Here is a comparison of TimesPro and competitor programs:")
            memory.chat_memory.add_ai_message(st.session_state.comparison_output)
            st.session_state.comparison_injected = True

        # Build ConversationalRetrievalChain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model_name=model_choice, openai_api_key=openai_key),
            retriever=retriever,
            memory=memory,
            return_source_documents=True
        )

        result = qa_chain.invoke({"question": user_question})
        st.write(f"💬 Answer: {result['answer']}")
# =========================
