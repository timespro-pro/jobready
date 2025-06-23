from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
import streamlit as st
import re


def _pretty_title(slug: str) -> str:
    """iim-calcutta-senior-management-programme → IIM Calcutta Senior Management Programme"""
    return re.sub(r"-", " ", slug).title()


def get_combined_response(
    pdf_text: str,
    url_texts: dict,
    timespro_url: str,
    competitor_url: str,
    model_choice: str = "gpt-4o",
) -> str:

    # 1️⃣  Build docs string (no truncation)
    docs = ""
    if pdf_text.strip():
        docs += "\n\n--- PDF Content ---\n" + pdf_text
    for i, (u, tx) in enumerate(url_texts.items(), 1):
        docs += f"\n\n--- URL {i} ({u}) Content ---\n{tx}"

    # 2️⃣  Make pretty names for the title line
    tp_name = _pretty_title(timespro_url.split("/")[-1])
    comp_name = _pretty_title(competitor_url.split("/")[-1])

    # 3️⃣  Strict prompt
    base_prompt = f"""
You are a strategic program analyst helping the sales team pitch a TimesPro program.
Using only the information in **Documents** below, generate the *ENTIRE* answer in **this exact Markdown skeleton**:

Sales-Enablement Brief: {tp_name} (TimesPro) vs {comp_name} (Competitor)

### Opening Summary Paragraph
<2–3 lines. Focus on strongest differentiators.>

### What Makes TimesPro’s Program Better
- **<Bold header (benefit)>** – Supporting explanation comparing to competitor  
- **<Bold header>** – …

### Who This Program Is Built For
TimesPro Program | Competitor Program
:-- | :--
✓ For professionals who want to … | ✓ For professionals who want to …
✓ For those seeking … | ✓ For those seeking …
✓ For aspirants targeting … | ✓ For aspirants targeting …
Curriculum Strength | Curriculum Limitation
✓ <TimesPro strength 1> | ✗ <Competitor limitation 1>
✓ <TimesPro strength 2> | ✗ <Competitor limitation 2>
✓ <TimesPro strength 3> | ✗ <Competitor limitation 3>

### 2 Taglines for Learner Interaction
**Aspirational (Phone call):** "<single sentence>"  
**Curriculum-led (Chat/email):** "<single sentence>"

### Price Justification & ROI *(Only include this section if TimesPro is more expensive)*
- <Reason 1>  
- <Reason 2>  
- <Reason 3>  
**Bottom line:** <one-liner that positions price as career-growth investment>

Tone rules:  
- No fluff. Concise, confident, benefit-driven.  
- No vague adjectives.  
- If unsure about any detail, OMIT it.  
- NEVER mention delivery platforms (Emeritus, upGrad, Coursera).

Documents:
{docs}
"""

    prompt = PromptTemplate.from_template(base_prompt)

    llm = ChatOpenAI(
        model_name=model_choice,
        temperature=0,
        openai_api_key=st.secrets["OPENAI_API_KEY"],
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({})
