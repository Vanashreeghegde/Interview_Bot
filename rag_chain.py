import os
import re
import random
from dotenv import load_dotenv
import google.generativeai as genai
from retriver import HybridRetriever

load_dotenv()

# ===============================
# Initialize Gemini
# ===============================

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash-lite")

retriever = HybridRetriever()

# ===============================
# GENERATION WRAPPER
# ===============================

def generate_response(prompt, temperature=0.4, max_tokens=500):
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens
        )
    )
    return response.text.strip()


# ==================================================
# GET NEXT QUESTION (REAL INTERVIEW ARCHITECTURE)
# ==================================================

def get_next_question(topic, difficulty, used_questions):

    # 🔥 Retrieval for grounding only
    query = f"{topic} production challenges system design tradeoffs failure cases"

    docs = retriever.retrieve(query)

    context = ""
    if docs:
        context = "\n\n".join([doc.page_content[:800] for doc in docs[:3]])

        # 🔥 Real fresher → intermediate interview generation prompt
    prompt = f"""
    You are a professional Data Science interviewer generating fresher-level to intermediate scenario-based questions.

    Topic: {topic}
    Difficulty: {difficulty} (1 = easy, 2 = intermediate)

    Rules:
    - Begin with basic questions suitable for freshers.
        Example questions at difficulty 1:
            - Why is linear regression used?
            - What is overfitting and how do you prevent it?
            - How would you handle missing values in a dataset?
    - Gradually increase difficulty depending on candidate answers.
    - Must be a complete, realistic scenario.
    - Must end with exactly ONE question mark.
    - Must be grammatically correct.
    - No repeated questions.
    - Avoid hallucinations.
    - Do NOT reference any text.
    - Output ONLY ONE question at a time.

    Now generate the next interview question for a fresher candidate.
    """

    question = generate_response(prompt, temperature=0.5, max_tokens=300)

    question = question.strip()

    if not question.endswith("?"):
        question += "?"

    # Strong duplicate detection
    for prev in used_questions:
        if question.lower()[:70] in prev.lower():
            return f"You are given a production problem related to {topic}. How would you systematically diagnose and solve it?"

    return question


# ==================================================
# EVALUATE ANSWER
# ==================================================

def evaluate_answer(question: str, answer: str):

    prompt = f"""
You are a strict technical interviewer.

Evaluate the candidate’s answer to this interview question.

Question:
{question}

Candidate Answer:
{answer}

Score from 0 to 10 based on:

- Technical accuracy
- Depth of reasoning
- Practical awareness
- Clarity of explanation

Return ONLY a single integer from 0 to 10.
No explanation.
"""

    response = generate_response(prompt, temperature=0.2, max_tokens=50)

    match = re.search(r'\b([0-9]|10)\b', response)
    score = int(match.group()) if match else 5

    return max(0, min(score, 10))


# ==================================================
# FINAL INTERVIEW FEEDBACK
# ==================================================

def generate_final_feedback(history, average_score, final_difficulty):

    transcript = ""

    for i, item in enumerate(history, 1):
        transcript += (
            f"Question {i}:\n{item['question']}\n\n"
            f"Answer:\n{item['answer']}\n\n"
            f"Score: {item['score']}/10\n"
            f"-------------------------\n"
        )

    prompt = f"""
You are a Senior Data Scientist conducting a final hiring review.

Below is the full interview transcript:

{transcript}

Average Score: {average_score}
Final Difficulty Level Reached: {final_difficulty}

Provide:

1. Overall performance summary
2. Strongest technical areas
3. Weakest areas
4. Recurring gaps
5. Clear improvement roadmap
6. Final overall score out of 10
7. Hiring decision:
   - Strong Hire
   - Hire
   - Borderline
   - Reject

Be realistic and professional.
"""

    return generate_response(prompt, temperature=0.5, max_tokens=800)