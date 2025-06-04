import streamlit as st
import os
import tempfile
import hashlib
from PyPDF2 import PdfReader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
#from utils.gcp_loader import load_vectorstore_from_gcp
from utils.load_vectorstore_from_gcp import load_vectorstore_from_gcp
from utils.upload_vectorstore_to_gcp import create_and_upload_vectorstore_from_pdf
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

gcp_config = {
    "bucket_name": os.getenv("GCP_BUCKET_NAME"),
    "prefix": os.getenv("GCP_VECTORSTORE_PREFIX"),
    "credentials": {
        "type": os.getenv("GCP_TYPE"),
        "project_id": os.getenv("GCP_PROJECT_ID"),
        "private_key_id": os.getenv("GCP_PRIVATE_KEY_ID"),
        "private_key": os.getenv("GCP_PRIVATE_KEY").replace("\\n", "\n"),
        "client_email": os.getenv("GCP_CLIENT_EMAIL"),
        "client_id": os.getenv("GCP_CLIENT_ID"),
        "auth_uri": os.getenv("GCP_AUTH_URI"),
        "token_uri": os.getenv("GCP_TOKEN_URI"),
        "auth_provider_x509_cert_url": os.getenv("GCP_AUTH_PROVIDER_CERT_URL"),
        "client_x509_cert_url": os.getenv("GCP_CLIENT_CERT_URL")
    }
}


def sanitize_url(url):
    return url.replace("https://", "").replace("http://", "").replace("/", "_")


def load_pdf(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    return "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])


st.set_page_config(page_title="Course Chatbot", layout="wide")
st.title("📘 Course Chatbot")

pdf_file = st.file_uploader("Upload a course brochure (PDF)", type="pdf")
selected_program = st.selectbox("Or select a TimesPro program", [""] + [
    "https://timespro.com/executive-education/iim-calcutta-executive-programme-in-business-management",
    "https://timespro.com/executive-education/iim-kozhikode-professional-certificate-programme-in-hr-management"
])
custom_url = st.text_input("Or manually enter a competitor program URL")

user_input = st.text_input("Ask a question about the course")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

vectorstore = None
retriever = None

if pdf_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        pdf_path = tmp.name

    pdf_text = load_pdf(pdf_path)
    pdf_hash = hashlib.md5(pdf_text.encode()).hexdigest()[:10]
    folder_name = f"uploaded_pdf_{pdf_hash}"
    vectorstore_path = f"{gcp_config['prefix']}/{folder_name}"

    with st.spinner("Processing uploaded PDF..."):
        try:
            vectorstore = load_vectorstore_from_gcp(
                bucket_name=gcp_config["bucket_name"],
                path=vectorstore_path,
                creds_dict=gcp_config["credentials"]
            )
            st.success("Vectorstore loaded from GCP.")
        except Exception:
            try:
                st.info("Vectorstore not found. Creating from PDF...")
                create_and_upload_vectorstore_from_pdf(
                    pdf_text=pdf_text,
                    bucket_name=gcp_config["bucket_name"],
                    folder_path=vectorstore_path,
                    creds_dict=gcp_config["credentials"]
                )
                vectorstore = load_vectorstore_from_gcp(
                    bucket_name=gcp_config["bucket_name"],
                    path=vectorstore_path,
                    creds_dict=gcp_config["credentials"]
                )
                st.success("Vectorstore created and loaded.")
            except Exception as e:
                st.error(f"Failed to process uploaded PDF: {e}")

elif selected_program:
    folder_name = f"timespro_com_executive_education_{sanitize_url(selected_program)}"
    vectorstore_path = f"{gcp_config['prefix']}/{folder_name}"

    with st.spinner("Loading TimesPro vectorstore..."):
        try:
            vectorstore = load_vectorstore_from_gcp(
                bucket_name=gcp_config["bucket_name"],
                path=vectorstore_path,
                creds_dict=gcp_config["credentials"]
            )
            st.success("Vectorstore loaded successfully.")
        except Exception as e:
            st.error(f"Vectorstore loading failed: {e}")

elif custom_url:
    folder_name = f"competitor_program_{sanitize_url(custom_url)}"
    vectorstore_path = f"{gcp_config['prefix']}/{folder_name}"

    with st.spinner("Loading competitor program vectorstore..."):
        try:
            vectorstore = load_vectorstore_from_gcp(
                bucket_name=gcp_config["bucket_name"],
                path=vectorstore_path,
                creds_dict=gcp_config["credentials"]
            )
            st.success("Vectorstore loaded successfully.")
        except Exception as e:
            st.error(f"Vectorstore loading failed: {e}")

if vectorstore:
    retriever = vectorstore.as_retriever()

    if user_input:
        llm = ChatOpenAI(temperature=0.2, openai_api_key=openai_api_key)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

        with st.spinner("Generating answer..."):
            result = qa_chain({"question": user_input, "chat_history": st.session_state.chat_history})
            st.session_state.chat_history.append((user_input, result["answer"]))
            st.markdown("**Answer:**")
            st.write(result["answer"])

    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### Chat History")
        for i, (q, a) in enumerate(st.session_state.chat_history):
            st.markdown(f"**Q{i+1}:** {q}")
            st.markdown(f"**A{i+1}:** {a}")
