import streamlit as st
import PyPDF2
import openai

openai.api_key = st.secrets["openai"]["api_key"]

def extract_text_from_pdf(pdf_file):
    """Extracts text from an uploaded PDF file."""
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

def generate_questions(job_description, job_id):
    """Generates 10 interview questions using GPT-4 based on the job description and job ID."""
    prompt = f"""
    You are a hiring manager creating interview questions for a job candidate.
    Based on the following job description and job ID, generate 10 relevant and thoughtful interview questions.
    
    ### Job Description:
    {job_description}
    
    ### Job ID:
    {job_id}
    
    The questions should assess technical skills, job-specific knowledge, and behavioral traits.
    
    Respond ONLY with a numbered list of 10 questions.
    """
    
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert recruiter generating job interview questions."},
            {"role": "user", "content": prompt}
        ]
    )
    
    questions = response.choices[0].message.content.strip()
    return questions.split("\n")  # Splitting numbered questions into a list

# Streamlit UI
st.title("AI Interview Question Generator")

st.write("Upload a job description and enter a Job ID to generate interview questions.")

job_desc_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])
job_id = st.text_input("Enter Job ID")

if job_desc_file and job_id:
    job_description = extract_text_from_pdf(job_desc_file)
    
    if st.button("Generate Interview Questions"):
        with st.spinner("Generating questions..."):
            questions = generate_questions(job_description, job_id)
            st.subheader("Generated Interview Questions:")
            for question in questions:
                st.write(question)
