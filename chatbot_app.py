import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

from utils.loaders import load_pdf, load_url_content
from utils.llm_chain import get_combined_response
from load_vectorstore_from_gcp import load_vectorstore_from_gcp

import tempfile

# ========== SECRETS & CREDENTIALS ==========
openai_key = st.secrets["OPENAI_API_KEY"]
gcp_credentials_dict = dict(st.secrets["GCP_SERVICE_ACCOUNT"])

gcp_config = {
    "bucket_name": "test_bucket_brian",
    "prefix": "vectorstores",
    "credentials": gcp_credentials_dict
}

# ========== STREAMLIT PAGE CONFIG ==========
st.set_page_config(page_title="AI Sales Assistant", layout="centered")
st.title("📚 AI Sales Assistant")

# ========== MODEL CHOICE ==========
model_choice = st.selectbox("Choose a model", ["gpt-3.5-turbo", "gpt-4", "gpt-4-1106-preview", "gpt-4o"])

# ========== FILE & URL INPUTS ==========
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
if selected_program == "-- Select a program --":
    selected_program = None

url_1 = selected_program
url_2 = st.text_input("Input Competitors Program URL")

# ========== SESSION STATE INIT ==========
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

if "additional_texts" not in st.session_state:
    st.session_state.additional_texts = ""

# ========== UTILITY ==========
def sanitize_url(url: str) -> str:
    return url.strip("/").split("/")[-1].replace("-", "_")

# ========== LOAD TimesPro VECTORSTORE ==========
retriever = None
if selected_program:
    folder_name = f"timespro_com_executive_education_{sanitize_url(selected_program)}"
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

# ========== COMPARISON + CLEAR ==========
col_compare, col_clear = st.columns([3, 1])

with col_compare:
    compare_clicked = st.button("Compare")

with col_clear:
    clear_clicked = st.button("Clear Cache 🧹", help="This will reset chat history and comparison")

if clear_clicked:
    st.session_state.clear()
    st.rerun()

# ========== PROCESS & STORE ADDITIONAL INPUTS ==========
if compare_clicked:
    if not (pdf_file or url_1 or url_2):
        st.warning("Please upload a PDF or enter at least one URL.")
    else:
        with st.spinner("Processing additional inputs..."):
            pdf_text = ""
            if pdf_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                    tmp_pdf.write(pdf_file.read())
                    pdf_path = tmp_pdf.name
                pdf_text = load_pdf(pdf_path)

            url_texts = load_url_content([url_1, url_2])
            combined_texts = "\n\n".join([pdf_text] + url_texts)

            st.session_state.additional_texts = combined_texts

            response = get_combined_response(pdf_text, url_texts, model_choice=model_choice)
            st.session_state.comparison_output = response
            st.session_state.comparison_injected = False
            st.success("Comparison complete:")
            st.write(response)

# ========== TEMPORARY VECTORSTORE FROM ADDITIONAL TEXT ==========
additional_retriever = None
if st.session_state.additional_texts.strip():
    docs = [Document(page_content=st.session_state.additional_texts)]
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    faiss_db = FAISS.from_documents(docs, embeddings)
    additional_retriever = faiss_db.as_retriever()

# ========== QA SECTION ==========
st.subheader("💬 Ask a follow-up question about the TimesPro or competitor programs")
user_question = st.text_input("Enter your question here")

if user_question and retriever:
    with st.spinner("Answering your question using all available context..."):

        if st.session_state.comparison_output and not st.session_state.comparison_injected:
            st.session_state.memory.chat_memory.add_user_message(
                "Here is a comparison of TimesPro and competitor programs:")
            st.session_state.memory.chat_memory.add_ai_message(st.session_state.comparison_output)
            st.session_state.comparison_injected = True

        all_retrievers = [retriever]
        if additional_retriever:
            all_retrievers.append(additional_retriever)

        class CombinedRetriever:
            def __init__(self, retrievers):
                self.retrievers = retrievers

            def get_relevant_documents(self, query):
                docs = []
                for r in self.retrievers:
                    docs.extend(r.get_relevant_documents(query))
                return docs

        combined_retriever = CombinedRetriever(all_retrievers)

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model_name=model_choice, openai_api_key=openai_key),
            retriever=combined_retriever,
            memory=st.session_state.memory,
            return_source_documents=True,
        )

        result = qa_chain.invoke({"question": user_question})
        st.write(f"💬 Answer: {result['answer']}")
