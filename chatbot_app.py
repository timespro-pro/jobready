import streamlit as st
import tempfile
from PyPDF2 import PdfReader
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import WebBaseLoader
from langchain.docstore.document import Document
from bs4 import BeautifulSoup
import requests

# === Session State Init ===
if "comparison_output" not in st.session_state:
    st.session_state.comparison_output = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# === Functions ===
def load_pdf(file_path):
    reader = PdfReader(file_path)
    texts = []
    for i, page in enumerate(reader.pages):
        content = page.extract_text()
        if content:
            texts.append(Document(page_content=content, metadata={"source": f"Page {i+1}"}))
    return texts

def create_vectorstore(docs, embeddings_model):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)
    return FAISS.from_documents(split_docs, embeddings_model)

def load_url_content(urls):
    combined_docs = []
    for i, url in enumerate(urls):
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = url
            combined_docs.extend(docs)
        except Exception as e:
            combined_docs.append(Document(page_content=f"[Error loading {url}: {e}]", metadata={"source": url}))
    return combined_docs

# === UI ===
st.title("📘 TimesPro: AI Sales Assistant (Internal Use Only)")
openai_key = st.text_input("🔑 Enter OpenAI API Key", type="password")
model_choice = st.selectbox("🤖 Model", ["gpt-3.5-turbo", "gpt-4"])
pdf_file = st.file_uploader("📄 Upload TimesPro Course Brochure (PDF)", type=["pdf"])
url_1 = st.text_input("🔗 TimesPro Course Page URL")
url_2 = st.text_input("🔗 Competitor Course Page URL")
compare_btn = st.button("🔍 Compare")

# === Load Inputs ===
pdf_docs, url_docs = [], []
if pdf_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        pdf_docs = load_pdf(tmp.name)

if url_1 or url_2:
    url_docs = load_url_content([url_1, url_2])

# === Comparison ===
if compare_btn and openai_key:
    llm = ChatOpenAI(model_name=model_choice, openai_api_key=openai_key)

    compare_prompt = PromptTemplate(
        input_variables=["pdf", "urls"],
        template="""
Compare the brochure content with the online pages. Highlight differences and similarities in:
- Fees
- Syllabus
- Duration
- Placement support
- EMI/loan options
- Course highlights

Brochure:
{pdf}

Online Pages:
{urls}
"""
    )

    compare_chain = LLMChain(llm=llm, prompt=compare_prompt)
    pdf_text = "\n".join([doc.page_content for doc in pdf_docs])
    url_text = "\n".join([doc.page_content for doc in url_docs])
    result = compare_chain.invoke({"pdf": pdf_text, "urls": url_text})
    st.session_state.comparison_output = result["text"]

st.markdown("### 🧾 Comparison Summary")
st.info(st.session_state.comparison_output or "No comparison yet.")

# === Chat Interface ===
user_question = st.text_input("💬 Ask about the TimesPro course:")

if user_question and pdf_docs and openai_key:
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    vectorstore = create_vectorstore(pdf_docs, embeddings)
    retriever = vectorstore.as_retriever()

    prompt_template = PromptTemplate(
        input_variables=["context", "question", "comparison_info"],
        template="""
Use only the provided context to answer the user’s question.
If not found, say: "The document does not contain this information."

Context:
{context}

Comparison Insights:
{comparison_info}

Question:
{question}
"""
    )

    llm = ChatOpenAI(model_name=model_choice, openai_api_key=openai_key)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=True
    )

    result = qa_chain.invoke({
        "question": user_question,
        "comparison_info": st.session_state.comparison_output
    })

    answer = result["result"]

    st.markdown("#### 💡 Answer:")
    st.write(answer)

    # === Source Display ===
    if result.get("source_documents"):
        st.markdown("#### 📄 Sources:")
        for i, doc in enumerate(result["source_documents"]):
            st.markdown(f"**Source {i+1}:** `{doc.metadata.get('source', 'unknown')}`")
            st.code(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)

    # === Fallback ===
    if "The document does not contain this information" in answer:
        st.markdown("🤖 Using fallback LLM...")

        full_context = "\n\n".join([
            st.session_state.comparison_output,
            "\n".join([doc.page_content for doc in pdf_docs]),
            "\n".join([doc.page_content for doc in url_docs])
        ])

        fallback_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a smart educational consultant.
Use the following context to answer the question.
If the information isn't available, say so clearly.

Context:
{context}

Question:
{question}
"""
        )

        fallback_chain = LLMChain(llm=llm, prompt=fallback_prompt)
        fallback_response = fallback_chain.invoke({
            "context": full_context,
            "question": user_question
        })

        st.markdown("#### 🧠 Fallback Answer:")
        st.write(fallback_response["text"])
