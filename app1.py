


Using the above code as as base
import streamlit as st
import PyPDF2
import requests
import boto3

# Load AWS credentials from Streamlit secrets
AWS_ACCESS_KEY = st.secrets["aws"]["AWS_ACCESS_KEY"]
AWS_SECRET_KEY = st.secrets["aws"]["AWS_SECRET_KEY"]
S3_BUCKET_NAME = "tensorflow-titans-bucket"  # Replace with your S3 bucket name
LAMBDA_URL = "YOUR_LAMBDA_FUNCTION_URL"  # Replace with your AWS Lambda endpoint
S3_FOLDER = "job_descriptions/"  # Folder in S3 bucket

# Initialize S3 Client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

# Function to upload file to S3
def upload_to_s3(file, filename):
    s3_key = S3_FOLDER + filename  # Store file in the job_descriptions folder
    s3_client.upload_fileobj(file, S3_BUCKET_NAME, s3_key)
    return f"s3://{S3_BUCKET_NAME}/{s3_key}"

# Streamlit UI
st.title("AI Interview Question Generator")

st.write("Upload a job description and enter additional details to generate interview questions.")

# Job Description Upload Button
job_desc_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])

# Additional input field
candidate_details = st.text_area("Enter job id from the HR portal ")

uploaded = False
s3_path = ""

if st.button("Upload to S3"):
    if job_desc_file and candidate_details:
        filename = job_desc_file.name
        with st.spinner("Uploading file..."):
            s3_path = upload_to_s3(job_desc_file, filename)
            uploaded = True
            st.success("File uploaded successfully!")
    else:
        st.error("Please upload a job description and enter a job ID before submitting.")

# Show Generate Questions button only after upload
if uploaded or s3_path:
    if st.button("Generate Interview Questions"):
        with st.spinner("Triggering AWS Lambda function..."):
            payload = {"s3_path": s3_path, "candidate_details": candidate_details}
            response = requests.post(LAMBDA_URL, json=payload)
            
            if response.status_code == 200:
                questions = response.json().get("questions", [])
                st.subheader("Generated Interview Questions:")
                for question in questions:
                    st.write(question)
            else:
                st.error("Error fetching questions from AWS Lambda.")

When generate questions are clicked, following code has to be triggered and shown in the panel 

###### Converts a job description into Interview questions #######

import boto3
import json

# Initialize the AWS Bedrock client
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",  # Change to your AWS region
)

def generate_interview_questions(job_description):
    """
    Sends a structured prompt to Claude 3 Sonnet on AWS Bedrock to generate interview questions
    and answers based on the job description and resume.
    """

    # Define the structured prompt
    prompt = f"""
    **Task:**
    You are an AI assistant skilled in analyzing job descriptions and resumes.
    - Extract the **Job Title** from the Job Description.
    - Extract the **total months of experience** from the resume.
    - Analyze the **Job Description (JD)** and extract **the top 5 most critical skills**.
    - Compare these skills with the **resume** to check matching/missing skills and give a **score from 1 to 10** for eligibility.
    - Generate **3 technical or conceptual questions** based on those extracted JD skills and candidate's experience and their answers.
      - For beginners: Simple definition-based questions.
      - For senior/mid-level roles: More advanced scenario-based questions.
    - Generate **2 project-based questions** from the resume related to past projects or experience that test the candidate’s real-world application of these skills along with their answers.

    **Job Description:**
    {job_description}

    **Resume:**


    **Output Format:**
    Just give the below mentioned fields only
    ```
    {{
      "JobTitle": "[Job Title extracted from the job description]",
      "Experience": "[Experience in Months extracted from resume]",
      "Score": "[Matching score between the resume and the job description]",
      "Questions": {{
        "Q1": "[Technical question 1]",
        "A1": "[Answer for Q1]",
        "Q2": "[Technical question 2]",
        "A2": "[Answer for Q2]",
        "Q3": "[Conceptual question 3]",
        "A3": "[Answer for Q3]",
        "Q4": "[Technical question 4]",
        "A4": "[Answer for Q4]",
        "Q5": "[Scenario-based question 5]",
        "A5": "[Answer for Q5]"
      }}
    }}
    ```
    """

    # Prepare the API request payload
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [
            {"role": "user", "content": prompt}  # Removed "system" role (only "user" is valid)
        ],
        "max_tokens": 700,
        "temperature": 0.0,
        "top_p": 0.9
    }

    # Invoke the Claude 3 Sonnet model using the Messages API
    response = bedrock.invoke_model(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        body=json.dumps(payload)
    )

    # Parse and display the response
    response_body = json.loads(response["body"].read().decode("utf-8"))

    return response_body["content"][0]["text"]


output = generate_interview_questions(job_description_text)
print(output)
 where the outputs are shown in the panel and then



and a save button is shown.
#######UUID table generated and interview questions uploaded to AWS bucket #####

import boto3
from botocore.exceptions import ClientError

def save_to_dynamodb(jdid,job_description, questions, AWS_ACCESS_KEY, AWS_SECRET_KEY, REGION_NAME):
    try:
        # Initialize DynamoDB client
        dynamodb = boto3.resource(
            'dynamodb',
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=REGION_NAME
        )

        # Get table
        table = dynamodb.Table('tensorflow_titans_user')

        # Check if the jdid already exists
        response = table.get_item(Key={'jdid': jdid})

        if 'Item' in response:
            # Append new role to existing item
            existing_item = response['Item']
            if 'roles' not in existing_item:
                existing_item['roles'] = []

            existing_item['roles'].append({
                'job_description': job_description,
                'questions': questions
            })

            # Update the table
            table.put_item(Item=existing_item)
        else:
            # Create a new item if jdid does not exist
            item = {
                'jdid': jdid,
          
                'roles': [{
                    'job_description': job_description,
                    'questions': questions
                }]
            }
            table.put_item(Item=item)

        print(f"Successfully saved/updated item with jdid: {jdid}")
        return {'jdid': jdid, 'status': 'success'}

    except ClientError as e:
        print(f"Error saving to DynamoDB: {str(e)}")
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise



After which the following code has to be triggered



Intergrate all the codes and show
