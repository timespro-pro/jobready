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

    **Output Format (JSON):**
    ```
    {{
      "JobTitle": "[Extracted Job Title]",
      "Experience": "[Experience in Months]",
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

    try:
        response = bedrock.invoke_model(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            body=json.dumps(payload)
        )
        
        # Debugging: Print raw response
        raw_response = response["body"].read().decode("utf-8")
        print("RAW RESPONSE FROM BEDROCK:", raw_response)  # Check for errors

        response_body = json.loads(raw_response)  # Ensure valid JSON
        
        # Check if "content" key exists
        if "content" in response_body and response_body["content"]:
            ai_text = response_body["content"][0]["text"]  # Get AI response text
            
            # Debugging: Print AI output before parsing
            print("AI OUTPUT:", ai_text)
            
            return json.loads(ai_text)  # Convert response text into a dictionary
        else:
            st.error("Bedrock returned an empty response. Please check the API call.")
            return {}

    except json.JSONDecodeError:
        st.error("Error: Received an invalid JSON response from AWS Bedrock.")
        return {}
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return {}
