import streamlit as st
from langchain_openai import ChatOpenAI
from utils.loaders import load_pdf, load_url_content
from utils.llm_chain import get_combined_response
import tempfile

# Set API key
openai_key = st.secrets["OPENAI_API_KEY"]

# Page config
st.set_page_config(page_title="PDF & Web Chatbot", layout="centered")
st.title("📚 Web + PDF Chatbot")

# Model selection dropdown
model_choice = st.selectbox(
    "Choose a model",
    ["gpt-3.5-turbo", "gpt-4", "gpt-4-1106-preview", "gpt-4o"],
    index=0
)

# Inputs
pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

# Dropdown for programs (replace with actual program names)
program_options = ["https://example.com/program1", "https://example.com/program2"]
selected_program = st.selectbox("Select Competitor Program URL", program_options)
url_1 = selected_program
url_2 = st.text_input("Input URL 2 (Optional)")

# Main comparison
if st.button("Compare"):
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

            # Generate the main comparison
            response = get_combined_response(pdf_text, url_texts, model_choice=model_choice)
            st.success("Here's the comparison:")
            st.write(response)

            # Follow-up chat input after showing comparison
            st.subheader("💬 Ask a follow-up question")
            user_question = st.text_input("Enter your question here")

            if user_question:
                with st.spinner("Answering your question..."):
                    followup_response = get_combined_response(
                        pdf_text,
                        url_texts,
                        model_choice=model_choice,
                        followup_question=user_question
                    )
                    st.write(f"💬 Answer: {followup_response}")
