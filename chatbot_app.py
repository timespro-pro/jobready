import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from utils.loaders import load_pdf, load_url_content
from utils.llm_chain import get_combined_response
from load_vectorstore_from_gcp import download_vectorstore_from_gcp, load_vectorstore
import tempfile

# ====== SECRETS ======
openai_key = st.secrets["OPENAI_API_KEY"]
gcp_credentials = st.secrets["gcp_service_account"]
# =====================

# ====== PAGE CONFIG ======
st.set_page_config(page_title="PDF & Web Chatbot", layout="centered")
st.title("📚 Web + PDF Chatbot")
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

# Program URL options
program_options = [
    "https://timespro.com/executive-education/iim-kozhikode-professional-certificate-programme-in-advanced-product-management",
    "https://example.com/program2"
]
selected_program = st.selectbox("Select TimesPro Program URL", program_options)
url_1 = selected_program
url_2 = st.text_input("Input URL 2 (Optional)")
# =====================

# ====== MAP URL TO GCP PREFIX ======
program_to_prefix = {
    "https://timespro.com/executive-education/iim-kozhikode-professional-certificate-programme-in-advanced-product-management": "timespro_vectors/kozhikode_program",
    "https://example.com/program2": "timespro_vectors/program2"
}
# ===================================

# ====== LOAD VECTORSTORE FROM GCP ======
with st.spinner("Loading TimesPro program details..."):
    selected_prefix = program_to_prefix.get(selected_program)

    if selected_prefix:
        local_path = download_vectorstore_from_gcp(
            bucket_name="test_bucket_brian",
            prefix=selected_prefix,
            gcp_credentials=gcp_credentials
        )
        vectorstore = load_vectorstore(local_path, openai_key)
        retriever = vectorstore.as_retriever()

        rag_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name=model_choice, openai_api_key=openai_key),
            retriever=retriever,
            chain_type="stuff"
        )
    else:
        st.error("No vectorstore available for the selected program.")
        rag_chain = None
# =======================================

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

            # Generate comparison from LLM
            response = get_combined_response(pdf_text, url_texts, model_choice=model_choice)
            st.success("Here's the comparison:")
            st.write(response)

            # Follow-up QA using vectorstore
            st.subheader("💬 Ask a follow-up question about the TimesPro program")
            user_question = st.text_input("Enter your question here")

            if user_question and rag_chain:
                with st.spinner("Answering your question using the TimesPro knowledge base..."):
                    answer = rag_chain.run(user_question)
                    st.write(f"💬 Answer: {answer}")
# ===============================
