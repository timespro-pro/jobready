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
import shutil

# === Session Initialization ===
if "comparison_output" not in st.session_state:
    st.session_state.comparison_output = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

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

# === UI ===
st.set_page_config(page_title="TimesPro Chatbot")
st.title("📘 TimesPro Course Advisor Chatbot")

if st.button("🧹 Clear Cache"):
    st.session_state.chat_history = []
    st.session_state.comparison_output = ""
    st.session_state.retriever = None
    st.session_state.pdf_text = ""
    st.success("Session cache cleared.")

programs = {
    "PG Diploma in Data Science": "data_science.pdf",
    "Executive MBA": "executive_mba.pdf",
    "Banking & Finance Certification": "banking_finance.pdf"
}

selected_program = st.selectbox("🎓 Select a TimesPro Program", list(programs.keys()))

# === Vectorstore loading ===
if selected_program:
    file_path = os.path.join("./vectorstore_pdfs", programs[selected_program])
    if os.path.exists(file_path):
        st.session_state.pdf_text = load_pdf(file_path)
        embeddings = OpenAIEmbeddings()
        vectorstore = create_vectorstore(st.session_state.pdf_text, embeddings)
        st.session_state.retriever = vectorstore.as_retriever()
        st.success(f"Vectorstore for '{selected_program}' loaded.")
    else:
        st.warning("PDF not found for selected program.")

url_1 = st.text_input("🔗 Enter TimesPro Course Page URL")
url_2 = st.text_input("🔗 Enter Competitor Course Page URL")
if st.button("🔍 Compare Course Info"):
    url_text = load_url_content([url_1, url_2])
    llm = ChatOpenAI()
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
    result = compare_chain.invoke({"pdf": st.session_state.pdf_text, "urls": url_text})
    st.session_state.comparison_output = result["text"]

if st.session_state.comparison_output:
    st.markdown("### 🧾 Comparison Summary")
    st.info(st.session_state.comparison_output)

user_question = st.text_input("💬 Ask your question about the TimesPro program:")

if user_question and st.session_state.retriever:
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
    prompt = PromptTemplate(input_variables=["context", "question", "comparison_info"], template=base_prompt)
    llm = ChatOpenAI()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=st.session_state.retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    response = qa_chain.invoke({
        "question": user_question,
        "comparison_info": st.session_state.comparison_output or "N/A"
    })

    answer = response["result"]
    st.markdown("#### 💡 Answer:")
    st.write(answer)

    if response.get("source_documents"):
        st.markdown("#### 📄 Sources Used:")
        for i, doc in enumerate(response["source_documents"]):
            st.markdown(f"**Source {i+1}:** {doc.metadata.get('source', 'unknown')}")
            st.code(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)

    if "The document does not contain this information" in answer:
        st.markdown("🤖 Attempting fallback with LLM reasoning...")
        url_text = load_url_content([url_1, url_2])
        fallback_context = "\n\n".join([st.session_state.comparison_output or "", st.session_state.pdf_text, url_text])
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
