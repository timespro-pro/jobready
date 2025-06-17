from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
import streamlit as st


def get_combined_response(
    pdf_text: str,
    url_texts: dict,
    timespro_url: str,
    competitor_url: str,
    model_choice: str = "gpt-3.5-turbo",
    followup_question: str | None = None,
) -> str:
    """
    Build a single, richly‑formatted prompt from PDF + URL extracts
    and return the LLM’s answer.
    """

    # ----------  assemble the document corpus  ----------
    docs = ""
    if pdf_text.strip():
        docs += "\n\n--- PDF Content ---\n" + pdf_text

    for i, (url, text) in enumerate(url_texts.items(), start=1):
        if text.strip():
            docs += f"\n\n--- URL {i} ({url}) Content ---\n{text}"

    # ----------  NEW sales‑enablement prompt  ----------
    base_prompt = f"""
You are a strategic program analyst helping the sales team pitch a TimesPro program to learners.
Based on the following documents, create a sales‑enablement brief using the exact structure below.

TimesPro's program: {timespro_url}
Competition's program: {competitor_url}

Task:
Create a sales‑enablement brief comparing the TimesPro program with the competitor’s program.

Output Format (follow strictly):

Opening Summary Paragraph (2–3 lines only):
Add a crisp, value‑led summary at the top of the brief. Highlight the strongest 1–2 differentiators.

What Makes TimesPro’s Program Better:
Provide 3‑4 bold, confident bullet points. Each bullet must have  
**Bold header** – a specific, career‑relevant benefit  
Supporting explanation – how it helps learners grow or lead better, and how it compares to the competitor’s program.

Who This Program Is Built For (Compare with Competitor – in table):
List 2‑3 audience points, plus 2‑3 curriculum‑strength comparisons.  
(Table columns: TimesPro Program | Competitor Program)

2 Taglines for Learner Interaction:
• One sentence that connects with learner aspirations.  
• One sentence that highlights a curriculum advantage.

Price Justification & ROI (only if TimesPro is more expensive):
• 2‑3 specific reasons the higher fee is justified – no generic claims.  
• Compare to competitor to show value for money.  
• Finish with a confident line positioning the price as a career‑growth investment.

Tone guidelines:
No fluff. Be concise, confident and benefit‑driven. Avoid vague adjectives. Focus on strategic, real‑world outcomes.  
If unsure about something, do **not** mention it.

Note:
Do **not** discuss the delivery platform when the competitor’s program is run via Emeritus, upGrad or Coursera.

Documents:
{docs}
"""

    # ----------  optional follow‑up  ----------
    if followup_question:
        base_prompt += (
            f"\n\nUser Follow‑Up Question:\n{followup_question}\nPlease answer this based on the comparison."
        )

    prompt = PromptTemplate.from_template(base_prompt)

    # ----------  run the chain  ----------
    openai_key = st.secrets["OPENAI_API_KEY"]
    llm = ChatOpenAI(
        model_name=model_choice,
        temperature=0.3,
        openai_api_key=openai_key,
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({})
