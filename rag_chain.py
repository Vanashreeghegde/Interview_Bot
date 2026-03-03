import os
import json
import re
from dotenv import load_dotenv
from groq import Groq
from retriver import HybridRetriever

load_dotenv()

# ===============================
# Initialize Groq
# ===============================

client = Groq(api_key=os.getenv("api"))

GENERATION_MODEL = "llama-3.1-8b-instant"
EVALUATION_MODEL = "llama-3.1-8b-instant"

retriever = HybridRetriever()


# ==================================================
# GENERATION WRAPPER
# ==================================================

def generate_response(prompt, model, temperature=0.4, max_tokens=500):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a professional Data Science interviewer."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


# ==================================================
# ADAPTIVE DIFFICULTY CONTROLLER
# ==================================================

def adjust_difficulty(current_difficulty, history):
    """
    Adaptive difficulty:
    1 = Easy
    2 = Medium
    3 = Hard
    """

    if not history:
        return current_difficulty

    recent_scores = [item["score"] for item in history[-3:]]
    avg_score = sum(recent_scores) / len(recent_scores)

    if avg_score >= 8 and current_difficulty < 3:
        return current_difficulty + 1

    if avg_score <= 4 and current_difficulty > 1:
        return current_difficulty - 1

    return current_difficulty


# ==================================================
# GET NEXT QUESTION (Phase + Adaptive Difficulty)
# ==================================================

def get_next_question(topic, difficulty, history, used_questions):

    question_count = len(history)

    # ---------------------------
    # PHASE CONTROL
    # ---------------------------
    if question_count < 2:
        phase = 1  # Fundamentals
    elif question_count < 5:
        phase = 2  # ML Case Studies
    else:
        phase = 3  # Coding Round

    # ---------------------------
    # ADJUST DIFFICULTY
    # ---------------------------
    difficulty = adjust_difficulty(difficulty, history)

    # ---------------------------
    # PHASE-BASED QUERY
    # ---------------------------
    if phase == 1:
        query = f"{topic} definition basics difference interview questions difficulty {difficulty}"
        instruction = "Ask a short conceptual question."
    elif phase == 2:
        query = f"{topic} real world case study production tradeoffs business problem difficulty {difficulty}"
        instruction = "Ask a realistic ML scenario or case study question."
    else:
        query = "SQL Python coding data manipulation debugging interview question"
        instruction = "Ask a coding question. It can be SQL or Python."

    docs = retriever.retrieve(query)

    if not docs:
        return f"What is the basic concept of {topic}?", difficulty

    context = "\n\n".join([doc.page_content[:800] for doc in docs[:3]])

    history_str = "\n".join(used_questions[-5:]) # Get last 5 questions

    # ---------------------------
    # STRICT GROUNDED PROMPT
    # ---------------------------
    prompt = f"""
You are a Data Science Interviewer. Your goal is to assess a candidate's 
knowledge of core Machine Learning,Data Science concepts,AI concepts,Gen AI concepts.Do not repeate the same question in different ways once done then it is done.Ask questions on fundament topics like list,tuple,dict
Include coding questions that exactly resembles like hacker rank coding questions.

Phase: {phase}
Difficulty Level: {difficulty}
Instruction: {instruction}

Rules:
- Start asking question in simple one line question from python then adapt the deficulty
- IGNORE basic HR questions or "what is an interview" questions.
- Focus on "Mid-Level" technical topics: 
   - Feature Engineering (Scaling, Encoding)
   - Model Evaluation (Precision, Recall, F1, ROC-AUC)
   - Standard Algorithms (Random Forest, Logistic Regression, K-Means)
   - Overfitting and Underfitting.
- Keep the questions practical and technical, but not overly academic.
- Stictly avoid asking single questions in different ways
- Use ONLY information from CONTEXT.
- Do not ask primary goal of data science interview
- Ask exactly ONE question.
- Must end with one question mark.
- No explanation.
- No repetition.
- No hallucination.

CONTEXT:
{context}
"""

    question = generate_response(
        prompt,
        model=GENERATION_MODEL,
        temperature=0.3,
        max_tokens=200
    ).strip()

    if not question.endswith("?"):
        question += "?"

    # Prevent repetition
    for prev in used_questions:
        if question.lower()[:60] in prev.lower():
            return f"Explain an important concept related to {topic}?", difficulty

    return question, difficulty


# ==================================================
# EVALUATE ANSWER (Structured Rubric)
# ==================================================

def evaluate_answer(question: str, answer: str):

    prompt = f"""
You are a strict technical interviewer.

Question:
{question}

Candidate Answer:
{answer}

Score using rubric:

Technical Accuracy (0–3)
Depth of Reasoning (0–3)
Practical Awareness (0–2)
Clarity (0–2)

Return STRICT JSON only:
{{
  "accuracy": int,
  "reasoning": int,
  "practical": int,
  "clarity": int
}}
"""

    response = generate_response(
        prompt,
        model=EVALUATION_MODEL,
        temperature=0.1,
        max_tokens=200
    )

    try:
        data = json.loads(response)
        total_score = (
            data["accuracy"] +
            data["reasoning"] +
            data["practical"] +
            data["clarity"]
        )
        return max(0, min(total_score, 10))
    except:
        match = re.search(r'\b([0-9]|10)\b', response)
        return int(match.group()) if match else 5


# ==================================================
# FINAL FEEDBACK
# ==================================================

def generate_final_feedback(history):

    if not history:
        return "No interview data available."

    avg_score = sum(item["score"] for item in history) / len(history)

    transcript = ""
    for i, item in enumerate(history, 1):
        transcript += (
            f"Question {i}:\n{item['question']}\n"
            f"Answer:\n{item['answer']}\n"
            f"Score: {item['score']}/10\n"
            f"-----------------------\n"
        )

    prompt = f"""
You are a Senior Data Scientist reviewing a candidate.

Transcript:
{transcript}

Average Score: {avg_score}

Provide:
1. Performance summary
2. Strongest areas
3. Weakest areas
4. Skill gaps
5. Improvement roadmap
6. Final score out of 10
7. Hiring decision

Be realistic.
"""

    return generate_response(
        prompt,
        model=EVALUATION_MODEL,
        temperature=0.4,
        max_tokens=800
    )
