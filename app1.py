import streamlit as st
import PyPDF2
import requests

# Function to extract text from an uploaded PDF file
def extract_text_from_pdf(pdf_file):
    """Extracts text from an uploaded PDF file."""
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

# Streamlit UI
st.title("AI Interview Question Generator")

st.write("Upload a job description and enter additional details to generate interview questions.")

# Job Description Upload Button
job_desc_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])

# Additional input field
candidate_details = st.text_area("Enter job id from the portal ")

# Button to trigger Lambda function
if job_desc_file:
    job_description = extract_text_from_pdf(job_desc_file)
    
    if st.button("Generate Interview Questions"):
        with st.spinner("Fetching questions from AWS Lambda..."):
            lambda_url = "YOUR_LAMBDA_FUNCTION_URL"  # Replace with your AWS Lambda endpoint
            payload = {"job_description": job_description, "candidate_details": candidate_details}
            response = requests.post(lambda_url, json=payload)
            
            if response.status_code == 200:
                questions = response.json().get("questions", [])
                st.subheader("Generated Interview Questions:")
                for question in questions:
                    st.write(question)
            else:
                st.error("Error fetching questions from AWS Lambda.")

