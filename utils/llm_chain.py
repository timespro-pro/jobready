from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
import streamlit as st

def get_combined_response(
    pdf_text: str,
    url_texts: dict,
    timespro_url: str,
    competitor_url: str,
    model_choice: str = "gpt-4o"
) -> str:
    prompt_template = """
You are a strategic program analyst helping the sales team pitch a TimesPro program to learners.

Using only the information provided in the below documents, create a sales‑enablement brief comparing the TimesPro program with the competitor’s program.

TimesPro’s program: {timespro_url}
Competitor’s program: {competitor_url}

Task:
Create the entire sales-enablement brief using the exact structure and Markdown format below:

Sales-Enablement Brief:  (TimesPro) vs (Competitor)

Opening Summary Paragraph
<2–3 lines only. Add a crisp, value-led summary at the top of the brief. Focus on the strongest 1–2 differentiators—such as CXO-readiness, curriculum strength, or ROI—that best position the TimesPro program.>

What Makes TimesPro’s Program Better
<Bold header (specific benefit)> – Supporting explanation comparing to competitor, focused on how it helps learners grow or lead better

<Bold header> – …

<Bold header> – …

<Bold header> – …

Who This Program Is Built For
Compare target audience and curriculum in the table format below.

TimesPro Program	Competitor Program
✓ For professionals who want to …	✓ For professionals who want to …
✓ For those seeking …	✓ For those seeking …
✓ For aspirants targeting …	✓ For aspirants targeting …
Curriculum Strength	Curriculum Limitation
✓ <TimesPro strength 1>	✗ <Competitor limitation 1>
✓ <TimesPro strength 2>	✗ <Competitor limitation 2>
✓ <TimesPro strength 3>	✗ <Competitor limitation 3>

2 Taglines for Learner Interaction
Aspirational (Phone call): "<Single sharp sentence that connects with the learner’s aspiration>"
Curriculum-led (Chat/email): "<One sentence that highlights a curriculum advantage>"

Price Justification & ROI (Include this section only if TimesPro is more expensive)
<Reason 1: Specific value-based justification>

<Reason 2>

<Reason 3>
Bottom line: <One-liner that positions the higher price as a career-growth investment>

Tone Rules:
No fluff. Be concise, confident, and benefit‑driven.
Avoid vague adjectives. Focus on strategic, real‑world outcomes.
If unsure about a detail, do NOT mention it.
NEVER mention delivery platforms (like Emeritus, upGrad, or Coursera), even if the competitor uses them.

--- TIMESPRO DOCUMENT ---
{tp_text}

--- COMPETITOR DOCUMENT ---
{comp_text}
"""

    tp_text = url_texts.get(timespro_url, "")
    comp_text = url_texts.get(competitor_url, "")

    full_prompt = PromptTemplate(
        input_variables=["timespro_url", "competitor_url", "tp_text", "comp_text"],
        template=prompt_template
    )

    chain = LLMChain(
        llm=ChatOpenAI(model_name=model_choice, openai_api_key=st.secrets["OPENAI_API_KEY"], temperature=0),
        prompt=full_prompt,
    )

    return chain.run({
        "timespro_url": timespro_url,
        "competitor_url": competitor_url,
        "tp_text": tp_text,
        "comp_text": comp_text
    })
