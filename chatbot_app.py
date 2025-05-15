import streamlit as st
from utils.loaders import load_pdf, load_url_content
from utils.llm_chain import get_combined_response
import tempfile

st.set_page_config(page_title="PDF & Web Chatbot", layout="centered")
st.title("📚 Web + PDF Chatbot")

pdf_file = st.file_uploader("Upload a PDF file", type="pdf")
url_1 = st.text_input("Input URL 1")
url_2 = st.text_input("Input URL 2")
question = st.text_area("Enter your question")

if st.button("View Output"):
    if not pdf_file or not (url_1 or url_2) or not question:
        st.warning("Please upload a PDF, enter at least one URL, and ask a question.")
    else:
        with st.spinner("Processing..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                tmp_pdf.write(pdf_file.read())
                pdf_path = tmp_pdf.name
            pdf_text = load_pdf(pdf_path)
            url_texts = load_url_content([url_1, url_2])
            response = get_combined_response(pdf_text, url_texts, question)
            st.success("Here's the result:")
            st.write(response)
