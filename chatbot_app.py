import streamlit as st
import tempfile
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from utils.loaders import load_pdf, load_url_content
from utils.llm_chain import get_combined_response
from load_vectorstore_from_gcp import load_vectorstore_from_gcp

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

# ====== MODEL SELECTION ======
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
url_2 = st.text_input("Input Competitor Program URL")

# ====== UTILS ======
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
if "comparison_output" not in st.session_state:
    st.session_state.comparison_output = ""

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# ====== COMPARISON LOGIC ======
col_compare, col_clear = st.columns([3, 1])
with col_compare:
    compare_clicked = st.button("Compare", disabled=(selected_program == "-- Select a program --"))
with col_clear:
    clear_clicked = st.button("Clear Cache 🧹")

if clear_clicked:
    st.session_state.clear()
    st.rerun()

if compare_clicked and (url_1 or url_2 or pdf_file):
    with st.spinner("Comparing TimesPro program with competitors..."):
        pdf_text = ""
        if pdf_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                tmp_pdf.write(pdf_file.read())
                pdf_path = tmp_pdf.name
            pdf_text = load_pdf(pdf_path)

        url_texts = load_url_content([url_1, url_2])
        comparison = get_combined_response(pdf_text, url_texts, model_choice=model_choice)
        st.session_state.comparison_output = comparison
        st.success("Comparison completed.")

# ====== DISPLAY COMPARISON RESULT ======
if st.session_state.get("comparison_output"):
    st.markdown("### 🤖 Comparison Output:")
    st.write(st.session_state.comparison_output)

# ====== CHATBOT QA SECTION ======
st.markdown("### 💬 Ask a follow-up question about the TimesPro program")
user_question = st.text_input("Enter your question here:")

if user_question and retriever:
    # ====== STRICT QA PROMPT TEMPLATE ======
    base_prompt = """
You are a helpful assistant. Answer the user’s question ONLY using the provided context from the TimesPro program documentation.
Do NOT use any prior knowledge or assumptions. If the answer is not found in the context, say “The document does not contain this information.”

Context:
{context}

Additional Notes (if any):
{comparison_info}

Question:
{question}
    """

    # NOTE: Only 'question' and 'comparison_info' should be input variables
    full_prompt = PromptTemplate(
        input_variables=["question", "comparison_info"],
        template=base_prompt
    )

    llm = ChatOpenAI(model_name=model_choice, openai_api_key=openai_key)

    st.session_state.qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": full_prompt},
        return_source_documents=True
    )

    comparison_context = st.session_state.comparison_output or "N/A"

    response = st.session_state.qa_chain.invoke({
        "question": user_question,
        "comparison_info": comparison_context
    })

    st.markdown("#### 💡 Answer:")
    st.write(response["result"])

    # ==== Source Documents ====
    if response.get("source_documents"):
        st.markdown("#### 📄 Sources Used:")
        for i, doc in enumerate(response["source_documents"]):
            st.markdown(f"**Source {i+1}:** `{doc.metadata.get('source', 'Unknown')}`")
            st.code(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)

    # ==== Fallback Logic ====
    if "The document does not contain this information" in response["result"]:
        st.markdown("🤖 Attempting fallback with LLM reasoning...")

        pdf_text = ""
        if pdf_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                tmp_pdf.write(pdf_file.read())
                pdf_path = tmp_pdf.name
            pdf_text = load_pdf(pdf_path)

        url_texts = load_url_content([url_1, url_2])
        fallback_context = "\n\n".join([st.session_state.comparison_output or "", pdf_text, url_texts])

        fallback_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are an expert educational consultant. Based on the context below, try to answer the user’s question.
If unsure, say you don’t know.

Context:
{context}

Question:
{question}
            """
        )

        fallback_chain = LLMChain(llm=llm, prompt=fallback_prompt)

        fallback_answer = fallback_chain.invoke({
            "context": fallback_context,
            "question": user_question
        })

        st.markdown("#### 🧠 Fallback Answer (LLM-based):")
        st.write(fallback_answer["text"])
