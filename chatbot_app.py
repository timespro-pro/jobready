import streamlit as st
import tempfile
import os
import json
from PyPDF2 import PdfReader
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup
import requests

# === Session init ===
if "comparison_output" not in st.session_state:
    st.session_state.comparison_output = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# === Functions ===
def load_pdf(file_path):
    reader = PdfReader(file_path)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

def create_vectorstore(text, embeddings_model):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents([text])
    return FAISS.from_documents(docs, embeddings_model)

def load_url_content(urls):
    combined_text = ""
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            combined_text += "\n".join([doc.page_content for doc in docs])
        except Exception as e:
            combined_text += f"\n[Error loading {url}: {e}]"
    return combined_text

def extract_text_from_html(url):
    try:
        html = requests.get(url, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text()
    except:
        return ""

# === UI ===
st.title("📘 TimesPro PDF & Competitor Analyzer Chatbot")
openai_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")
model_choice = st.selectbox("🤖 Select Model", ["gpt-3.5-turbo", "gpt-4"])
pdf_file = st.file_uploader("📄 Upload TimesPro Course Brochure (PDF)", type=["pdf"])
url_1 = st.text_input("🔗 Enter TimesPro Course Page URL")
url_2 = st.text_input("🔗 Enter Competitor Course Page URL")
compare_btn = st.button("🔍 Compare Course Info")

# === Comparison logic ===
if compare_btn and openai_key:
    raw_text = ""
    if pdf_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.read())
            tmp_path = tmp.name
        raw_text = load_pdf(tmp_path)

    url_text = load_url_content([url_1, url_2])
    llm = ChatOpenAI(model_name=model_choice, openai_api_key=openai_key)
    compare_prompt = PromptTemplate(
        input_variables=["pdf", "urls"],
        template="""
You are an educational content analyst. Compare the course described in the brochure (below)
with the two online course pages.

Brochure Content:
{pdf}

Online Page Content:
{urls}

Compare details such as fees, syllabus, duration, placement support, loan/EMI options, and course highlights.
Summarize the similarities and differences clearly.
        """
    )
    compare_chain = LLMChain(llm=llm, prompt=compare_prompt)
    comparison_result = compare_chain.invoke({"pdf": raw_text, "urls": url_text})
    st.session_state.comparison_output = comparison_result["text"]

st.markdown("### 🧾 Comparison Summary")
st.info(st.session_state.comparison_output or "No comparison done yet.")

# === Vectorstore for PDF ===
retriever = None
if pdf_file and openai_key:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        tmp_path = tmp.name
    pdf_text = load_pdf(tmp_path)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    vectorstore = create_vectorstore(pdf_text, embeddings)
    retriever = vectorstore.as_retriever()

# === Chat interface ===
user_question = st.text_input("💬 Ask your question about the TimesPro program:")

if user_question and retriever:

    # === Custom prompt with context and comparison ===
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
    full_prompt = PromptTemplate(
        input_variables=["context", "question", "comparison_info"],
        template=base_prompt
    )

    llm = ChatOpenAI(model_name=model_choice, openai_api_key=openai_key)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": full_prompt},
        return_source_documents=True
    )

    comparison_context = st.session_state.comparison_output or "N/A"

    response = qa_chain.invoke({
        "question": user_question,
        "comparison_info": comparison_context
    })

    answer = response["result"]

    st.markdown("#### 💡 Answer:")
    st.write(answer)

    # === Source document display ===
    if response.get("source_documents"):
        st.markdown("#### 📄 Sources Used:")
        for i, doc in enumerate(response["source_documents"]):
            st.markdown(f"**Source {i+1}:** {doc.metadata.get('source', 'unknown')}")
            st.code(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)

    # === Fallback logic ===
    if "The document does not contain this information" in answer:
        st.markdown("🤖 Attempting fallback with LLM reasoning...")

        url_text = load_url_content([url_1, url_2])
        fallback_context = "\n\n".join([st.session_state.comparison_output or "", pdf_text, url_text])

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
