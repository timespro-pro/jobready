from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
import streamlit as st

def get_combined_response(pdf_text, url_texts, question):
    content = ""

    if pdf_text.strip():
        content += "\n\n--- PDF Content ---\n" + pdf_text

    for i, text in enumerate(url_texts):
        if text.strip():
            content += f"\n\n--- URL {i+1} Content ---\n" + text

    # Use structured prompt for sales-enablement brief
    prompt_template = """
You are a strategic program analyst helping counselors pitch a TimesPro program to learners. Based on the following documents, create a sales-enablement brief using the exact structure below.

Documents:
{docs}

Question/Task:
{question}

Output Format (follow strictly):

What Makes TimesPro’s Program Better
Provide 5–6 bold, confident bullet points. Each bullet should have:
Bold header: Specific, career-relevant benefit  
Supporting explanation: Show how it helps learners grow or lead better, and how it compares to the competitor’s program.

Who This Program Is Built For (Compare with Competitor – in table)
✓ For professionals who want to…  
✓ For those seeking…  
✓ For aspirants targeting…  
(Table format: TimesPro Program | Competitor Program)

Tagline for Learner Interaction
One sharp, sales-friendly sentence a counselor can say on a call.

Price Justification (Include only if TimesPro is more expensive)
- 2–3 crisp bullets justifying the higher fee.  
- End with a confident one-liner that explains the value.

Tone:  
No fluff. Be concise, confident, and benefit-driven. No vague adjectives. Focus on strategic, real-world outcomes.
"""

    prompt = PromptTemplate.from_template(prompt_template)

    openai_key = st.secrets["OPENAI_API_KEY"]
    llm = ChatOpenAI(temperature=0.3, model_name="gpt-3.5-turbo", openai_api_key=openai_key)

    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run({"docs": content, "question": question})
    return response
