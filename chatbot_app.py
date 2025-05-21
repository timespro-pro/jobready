import streamlit as st
#from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from utils.loaders import load_pdf, load_url_content
from utils.llm_chain import get_combined_response
import tempfile

openai_key = st.secrets["OPENAI_API_KEY"]
llm = ChatOpenAI(temperature=0.3, model_name="gpt-3.5-turbo", openai_api_key=openai_key)

st.set_page_config(page_title="PDF & Web Chatbot", layout="centered")
st.title("📚 Web + PDF Chatbot")

# Inputs
pdf_file = st.file_uploader("Upload a PDF file", type="pdf")
url_1 = st.text_input("Input URL 1")
url_2 = st.text_input("Input URL 2")
question = st.text_area("Enter your question", height=150)

# Process
if st.button("View Output"):
    if not (pdf_file or url_1 or url_2):
        st.warning("Please upload a PDF or enter at least one URL.")
    elif not question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Processing..."):
            pdf_text = ""
            if pdf_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                    tmp_pdf.write(pdf_file.read())
                    pdf_path = tmp_pdf.name
                pdf_text = load_pdf(pdf_path)

            url_texts = load_url_content([url_1, url_2])
            response = get_combined_response(pdf_text, url_texts, question)
            st.success("Here's the result:")
            st.write(response)
