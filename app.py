import os
import streamlit as st
from dotenv import load_dotenv
from utils.pdf_utils import extract_text_from_pdf
from utils.web_utils import extract_text_from_url
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import Document
from langchain.text_splitter import CharacterTextSplitter

load_dotenv("credentials/.env")

st.set_page_config(page_title="PDF & Web LLM Compare", layout="centered")
st.title("📄🔍 PDF vs Web Comparison using LLM")

pdf_file = st.file_uploader("Upload a PDF File", type="pdf")
web_url = st.text_input("WebSearch - Enter URL")

if st.button("Compare"):
    if not pdf_file or not web_url:
        st.error("Please upload a PDF and enter a URL.")
    else:
        with st.spinner("Processing..."):
            pdf_text = extract_text_from_pdf(pdf_file)
            web_text = extract_text_from_url(web_url)

            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            documents = text_splitter.create_documents([pdf_text, web_text])

            llm = ChatOpenAI(temperature=0)
            chain = load_qa_chain(llm, chain_type="stuff")
            result = chain.run(input_documents=documents, question="Compare the PDF and Web content.")

        st.subheader("🧠 LLM Result")
        st.write(result)
