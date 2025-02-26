import streamlit as st
import boto3
import json
import PyPDF2
import uuid
from botocore.exceptions import NoCredentialsError, ClientError

# Load AWS Credentials from Streamlit Secrets
AWS_ACCESS_KEY = st.secrets["aws"]["AWS_ACCESS_KEY"]
AWS_SECRET_KEY = st.secrets["aws"]["AWS_SECRET_KEY"]
REGION_NAME = st.secrets["aws"]["REGION_NAME"]
DYNAMODB_TABLE = "InterviewQuestions"

# Initialize AWS Clients
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
table = dynamodb.Table(DYNAMODB_TABLE)

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

# Function to generate interview questions using Claude 3
def generate_interview_questions(job_description):
    prompt = f"""
    You are an AI assistant skilled in analyzing job descriptions.
    - Extract the **Job Title** from the Job Description.
    - Identify the **top 5 critical skills** from the JD.
    - Generate **5 technical/conceptual questions** and **5 project-based questions** with answers.

    **Job Description:**
    {job_description}

    **Output Format:**
    ```json
    {{
      "JobTitle": "[Extracted Job Title]",
      "Skills": ["Skill1", "Skill2", "Skill3", "Skill4", "Skill5"],
      "Questions": {{
        "Q1": "[Question 1]", "A1": "[Answer 1]",
        "Q2": "[Question 2]", "A2": "[Answer 2]",
        "Q3": "[Question 3]", "A3": "[Answer 3]",
        "Q4": "[Question 4]", "A4": "[Answer 4]",
        "Q5": "[Question 5]", "A5": "[Answer 5]",
        "Q6": "[Project Question 6]", "A6": "[Answer 6]",
        "Q7": "[Project Question 7]", "A7": "[Answer 7]",
        "Q8": "[Project Question 8]", "A8": "[Answer 8]",
        "Q9": "[Project Question 9]", "A9": "[Answer 9]",
        "Q10": "[Project Question 10]", "A10": "[Answer 10]"
      }}
    }}
    ```
    """

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 800,
        "temperature": 0.5,
        "top_p": 0.9
    }

    try:
        response = bedrock.invoke_model(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            body=json.dumps(payload)
        )
        response_body = json.loads(response["body"].read().decode("utf-8"))
        output_text = response_body["content"][0]["text"]
        return json.loads(output_text)
    except Exception as e:
        st.error(f"Error generating interview questions: {str(e)}")
        return None

# Function to upload data to DynamoDB
def upload_to_dynamodb(job_id, job_description, questions):
    data = {
        "JobID": job_id,
        "JobDescription": job_description,
        "Questions": questions
    }
    try:
        table.put_item(Item=data)
        return f"Data successfully uploaded to DynamoDB with Job ID: {job_id}"
    except NoCredentialsError:
        return "AWS Credentials not found."
    except ClientError as e:
        return f"Error uploading to DynamoDB: {e}"

# Streamlit UI
st.title("AI-Powered Interview Question Generator")

# File Uploader
job_desc_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])

if job_desc_file:
    job_description = extract_text_from_pdf(job_desc_file)
    job_id = str(uuid.uuid4())  # Generate a unique Job ID

    if st.button("Generate Interview Questions"):
        with st.spinner("Generating questions..."):
            generated_data = generate_interview_questions(job_description)
            
            if generated_data:
                st.subheader(f"Generated Questions for Job ID: {job_id}")
                st.write(f"**Job Title:** {generated_data['JobTitle']}")
                st.write("**Critical Skills:**", ", ".join(generated_data['Skills']))
                
                for i in range(1, 11):
                    st.markdown(f"**Q{i}:** {generated_data['Questions'][f'Q{i}']}")
                    st.write(f"**A{i}:** {generated_data['Questions'][f'A{i}']}")

                if st.button("Upload to DynamoDB"):
                    with st.spinner("Uploading..."):
                        db_result = upload_to_dynamodb(job_id, job_description, generated_data["Questions"])
                        st.success(db_result)
else:
    st.warning("Please upload a job description to proceed.")
