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

def generate_questions(job_description, resume_text):
    """Generates 10 interview questions using GPT-4 based on the job description and resume."""
    prompt = f"""
    You are a hiring manager creating interview questions for a job candidate.
    Based on the following job description and candidate's resume, generate 10 relevant and thoughtful interview questions.
    
    ### Job Description:
    {job_description}
    
    ### Candidate's Resume:
    {resume_text}
    
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

st.write("Upload a job description and a candidate's resume to generate interview questions.")

job_desc_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])
resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if job_desc_file and resume_file:
    job_description = extract_text_from_pdf(job_desc_file)
    resume_text = extract_text_from_pdf(resume_file)
    
    if st.button("Generate Interview Questions"):
        with st.spinner("Generating questions..."):
            questions = generate_questions(job_description, resume_text)
            st.subheader("Generated Interview Questions:")
            for question in questions:
                st.write(question)
