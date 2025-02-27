import streamlit as st
import PyPDF2
import boto3
import json

# Load AWS credentials from Streamlit secrets
AWS_ACCESS_KEY = st.secrets["aws"]["AWS_ACCESS_KEY"]
AWS_SECRET_KEY = st.secrets["aws"]["AWS_SECRET_KEY"]

# Initialize AWS Bedrock Client
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

MAX_JOB_DESC_LENGTH = 4000  # Adjust based on token limits


def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF."""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text[:MAX_JOB_DESC_LENGTH]  # Trim to token limit


def generate_interview_questions(job_description):
    """
    Sends a structured prompt to Claude 3 Sonnet on AWS Bedrock to generate interview questions.
    """
    prompt = f"""
    **Task:**
    You are an AI assistant skilled in analyzing job descriptions and resumes.
    - Extract the **Job Title** from the Job Description.
    - Extract the **total months of experience** from the resume.
    - Analyze the **Job Description (JD)** and extract **the top 5 most critical skills**.
    - Generate **3 technical or conceptual questions** based on those extracted JD skills.
      - For beginners: Simple definition-based questions.
      - For senior/mid-level roles: More advanced scenario-based questions.
    - Generate **2 project-based questions** related to past projects or experience that test real-world application of these skills.

    **Job Description:**
    {job_description}

    **Output Format:**
    ```
    {{
      "JobTitle": "[Job Title extracted from the job description]",
      "Experience": "[Experience in Months extracted from resume]",
      "Questions": {{
        "Q1": "[Technical question 1]",
        "Q2": "[Technical question 2]",
        "Q3": "[Conceptual question 3]",
        "Q4": "[Project-based question 4]",
        "Q5": "[Project-based question 5]"
      }}
    }}
    ```
    """

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 700,
        "temperature": 0.0,
        "top_p": 0.9
    }

    response = bedrock.invoke_model(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        body=json.dumps(payload)
    )

    response_body = json.loads(response["body"].read().decode("utf-8"))
    return json.loads(response_body["content"][0]["text"])


# Streamlit UI
st.title("AI Interview Question Generator")

st.write("Upload a job description **OR** enter it manually.")

# Job Description Upload (Optional)
job_desc_file = st.file_uploader("Upload Job Description (PDF) (Optional)", type=["pdf"])

# Manual Job Description Entry
job_description_text = st.text_area("Or, enter job description manually", "")

# Handle PDF Upload
if job_desc_file is not None:
    with st.spinner("Extracting text from PDF..."):
        extracted_text = extract_text_from_pdf(job_desc_file)
        job_description_text = extracted_text  # Auto-fill the text area

# Button to generate questions
if st.button("Generate Interview Questions"):
    if not job_description_text.strip():
        st.error("Please provide a job description either by uploading a PDF or entering text manually.")
    else:
        with st.spinner("Generating interview questions..."):
            questions_data = generate_interview_questions(job_description_text)
            
            st.subheader(f"Job Title: {questions_data.get('JobTitle', 'N/A')}")
            st.write(f"Experience Required: {questions_data.get('Experience', 'N/A')} months")
            
            st.subheader("Generated Interview Questions:")
            for i in range(1, 6):
                question = questions_data.get("Questions", {}).get(f"Q{i}", "")
                if question:
                    st.write(f"**Q{i}:** {question}")
                    st.markdown("---")
