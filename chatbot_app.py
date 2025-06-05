import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from utils.loaders import load_pdf, load_url_content
from utils.llm_chain import get_combined_response
from load_vectorstore_from_gcp import load_vectorstore_from_gcp
import tempfile

# LangChain imports for embeddings, text splitter, FAISS, etc.
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import BaseRetriever

# ====== SECRETS & GCP CREDENTIALS ======
openai_key = st.secrets["OPENAI_API_KEY"]
gcp_credentials_dict = dict(st.secrets["GCP_SERVICE_ACCOUNT"])

gcp_config = {
    "bucket_name": "test_bucket_brian",
    "prefix": "vectorstores",
    "credentials": gcp_credentials_dict
}
# =======================================

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

# ====== SANITIZE URL ======
def sanitize_url(url: str) -> str:
    return url.strip("/").split("/")[-1].replace("-", "_")

folder_name = f"timespro_com_executive_education_{sanitize_url(selected_program)}"

# ====== LOAD TIMESPRO VECTORSTORE ======
with st.spinner("Loading TimesPro program details..."):
    try:
        vectorstore_path = f"{gcp_config['prefix']}/{folder_name}"
        vectorstore = load_vectorstore_from_gcp(
            bucket_name=gcp_config["bucket_name"],
            path=vectorstore_path,
            creds_dict=gcp_config["credentials"]
        )
        timespro_retriever = vectorstore.as_retriever()
        st.success("TimesPro vectorstore loaded successfully.")
    except Exception as e:
        st.error(f"Vectorstore loading failed: {e}")
        timespro_retriever = None

# ====== CONVERSATION MEMORY INIT ======
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

# ====== COMPARISON & CLEAR CACHE BUTTONS SIDE BY SIDE ======
col_compare, col_clear = st.columns([3, 1])

with col_compare:
    compare_clicked = st.button("Compare")

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
                st.experimental_rerun()
        with col_cancel:
            if st.button("Cancel", key="cancel_clear"):
                st.session_state.show_confirm_clear = False

# ====== HANDLE COMPARISON LOGIC ======
if compare_clicked:
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

# ====== COMBINED RETRIEVER CLASS ======
class CombinedRetriever(BaseRetriever):
    def __init__(self, retrievers):
        self.retrievers = retrievers

    def get_relevant_documents(self, query):
        results = []
        for retriever in self.retrievers:
            results.extend(retriever.get_relevant_documents(query))
        # Optionally, you can sort/filter results here if needed
        return results

# ====== QA CHATBOT SECTION ======
st.subheader("💬 Ask a follow-up question about the TimesPro program")
user_question = st.text_input("Enter your question here")

if user_question and timespro_retriever:
    with st.spinner("Answering your question using the knowledge base..."):

        # Build vectorstore from PDF + competitor URLs dynamically if content present
        additional_retrievers = []
        embedding_model = OpenAIEmbeddings(openai_api_key=openai_key)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

        # Prepare docs list from PDF and URLs
        docs = []

        if pdf_file:
            # Load PDF text if not already loaded in this session
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                tmp_pdf.write(pdf_file.read())
                pdf_path = tmp_pdf.name
            pdf_text = load_pdf(pdf_path)
            if pdf_text.strip():
                docs.append(pdf_text)

        # Load competitor URLs content
        if url_1 or url_2:
            url_texts = load_url_content([url_1, url_2])
            for txt in url_texts:
                if txt.strip():
                    docs.append(txt)

        # If we have docs from PDF/URLs, create a vectorstore retriever for them
        if docs:
            chunks = []
            for doc_text in docs:
                chunks.extend(text_splitter.split_text(doc_text))

            if chunks:
                pdf_url_vectorstore = FAISS.from_texts(chunks, embedding_model)
                additional_retrievers.append(pdf_url_vectorstore.as_retriever())

        # Combine TimesPro retriever with additional retrievers (if any)
        combined_retriever = (
            CombinedRetriever([timespro_retriever] + additional_retrievers)
            if additional_retrievers
            else timespro_retriever
        )

        # Inject comparison text once into memory if available
        if st.session_state.comparison_output and not st.session_state.comparison_injected:
            st.session_state.memory.chat_memory.add_user_message(
                "Here is a comparison of TimesPro and competitor programs:"
            )
            st.session_state.memory.chat_memory.add_ai_message(st.session_state.comparison_output)
            st.session_state.comparison_injected = True

        # Build the ConversationalRetrievalChain with combined retriever
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model_name=model_choice, openai_api_key=openai_key),
            retriever=combined_retriever,
            memory=st.session_state.memory,
            return_source_documents=True,
        )

        result = qa_chain.invoke({"question": user_question})
        st.write(f"💬 Answer: {result['answer']}")
