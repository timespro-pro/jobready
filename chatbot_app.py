import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from utils.loaders import load_pdf, load_url_content
from utils.llm_chain import get_combined_response
from load_vectorstore_from_gcp import load_vectorstore_from_gcp
import tempfile
import os
 
# ====== SECRETS & GCP CREDENTIALS ======
openai_key = st.secrets["OPENAI_API_KEY"]
gcp_credentials_dict = dict(st.secrets["GCP_SERVICE_ACCOUNT"])
 
gcp_config = {
    "bucket_name": "test_bucket_brian",
    "prefix": "vectorstores",
    "credentials": gcp_credentials_dict  # Pass as dict to match loader expectations
}
# ======================================
 
st.set_page_config(page_title="AI Sales Assistant", layout="centered")
st.title("📚 AI Sales Assistant")

# ====== CLEAR CACHE BUTTON - CLEANED UI & CONFIRMATION ======
st.markdown("""
    <style>
    .clear-btn-container {
        display: flex;
        justify-content: flex-start;
        margin-top: -40px;
        margin-bottom: 10px;
    }
    .clear-btn button {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        padding: 0.25rem 1rem;
        border-radius: 0.4rem;
        white-space: nowrap;
    }
    </style>
""", unsafe_allow_html=True)

clear_btn_col, _ = st.columns([1, 9])
with clear_btn_col:
    with st.container():
        if st.button("Clear Cache", help="Click to clear memory and chat history", key="clear_cache_btn"):
            st.session_state.show_confirm = True

# ====== CONFIRMATION PROMPT ======
if st.session_state.get("show_confirm", False):
    confirm_box = st.empty()
    with confirm_box.container():
        st.warning("⚠️ Are you sure you want to clear memory and reset chat? This action cannot be undone.")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("✅ Yes, Clear", key="confirm_clear_yes"):
                st.session_state.memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    input_key="question",
                    output_key="answer",
                    return_messages=True,
                    k=7
                )
                st.session_state.comparison_output = ""
                st.session_state.comparison_injected = False
                st.session_state.show_confirm = False
                confirm_box.empty()
                st.rerun()
        with col2:
            if st.button("❌ Cancel", key="confirm_clear_no"):
                st.session_state.show_confirm = False
                confirm_box.empty()


 
# ====== MODEL CHOICE ======
model_choice = st.selectbox(
    "Choose a model",
    ["gpt-3.5-turbo", "gpt-4", "gpt-4-1106-preview", "gpt-4o"],
    index=0
)
 
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
url_2 = st.text_input("Input Competitors Program URL")
 
# Sanitize URL
def sanitize_url(url: str) -> str:
    return url.strip("/").split("/")[-1].replace("-", "_")
 
folder_name = f"timespro_com_executive_education_{sanitize_url(selected_program)}"
 
# ====== LOAD VECTORSTORE ======
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
        retriever = None
 
# ====== CONVERSATION MEMORY ======
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True,
        k=7
    )
 
# ====== SESSION STATE INIT ======
if "comparison_output" not in st.session_state:
    st.session_state.comparison_output = ""
 
if "comparison_injected" not in st.session_state:
    st.session_state.comparison_injected = False
 
# ====== MAIN COMPARISON BUTTON ======
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
            st.session_state.comparison_injected = False
            st.success("Here's the comparison:")
            st.write(response)
 
# ====== QA CHATBOT SECTION ======
st.subheader("💬 Ask a follow-up question about the TimesPro program")
user_question = st.text_input("Enter your question here")
 
if user_question and retriever:
    with st.spinner("Answering your question using the knowledge base..."):
 
        # Inject comparison output to memory once
        if st.session_state.comparison_output and not st.session_state.comparison_injected:
            st.session_state.memory.chat_memory.add_user_message(
                "Here is a comparison of TimesPro and competitor programs:")
            st.session_state.memory.chat_memory.add_ai_message(st.session_state.comparison_output)
            st.session_state.comparison_injected = True
 
        # Build ConversationalRetrievalChain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model_name=model_choice, openai_api_key=openai_key),
            retriever=retriever,
            memory=st.session_state.memory,
            return_source_documents=True,
        )
 
        result = qa_chain.invoke({"question": user_question})
        st.write(f"💬 Answer: {result['answer']}")
