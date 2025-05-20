from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI

def get_combined_response(pdf_text, url_texts, question):
    # Combine PDF and URL content
    content = "\n\n--- PDF Content ---\n" + pdf_text
    for i, text in enumerate(url_texts):
        content += f"\n\n--- URL {i+1} Content ---\n" + text

    # Custom prompt template for sales-enablement comparison
    prompt = PromptTemplate.from_template(
        """I want to create a sales-enablement brief that shows why the following TimesPro program is stronger than a competing offering, in a confident, concise format for counselors to use in learner conversations.

TimesPro Program: https://timespro.com/executive-education/iim-kozhikode-professional-certificate-programme-in-advanced-product-management  
Price: 1.68L  
Competing Program: https://iimkozhikode.emeritus.org/iimk-professional-certificate-programme-in-product-management  
Price: 1.9L  

Given the following documents:
{docs}

Answer the question: {question}

Output Format (very important — follow this structure exactly):

What Makes TimesPro’s Program Better  
Provide 5–6 bold, confident bullet points that clearly show where the TimesPro program wins. Use this format:  
Bold header: Specific, career-relevant benefit  
Supporting explanation: No fluff, show how it helps the learner grow or lead better and compares with the competitors' program in table.

Who This Program Is Built For (compare with the competition and put it in a table)  
Use 2–5 learner goal statements. Format them as:  
✓ For professionals who want to…  
✓ For those seeking…  
✓ For aspirants targeting…

Tagline for Learner Interaction  
One sharp, sales-friendly line that a counselor can say in a pitch or call.  
E.g.: “[competitor's program] teaches you to lead people & teams. [TimesPro's program] prepares you to lead businesses.”

Price Justification (If TimesPro Program Costs More, else skip this and add it as a plus point for TimesPro in an earlier point)  
Give 2–3 concise bullets on why the higher fee is justified.  
Then give a confident one-liner for the price.

Tone:  
No long paragraphs.  
Use confident, sales-ready language.  
Avoid adjectives like “renowned” or “popular” — focus on real, strategic learner benefits."""
    )

    llm = OpenAI(temperature=0.3, model_name="gpt-3.5-turbo")
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run({"docs": content, "question": question})
    return response
