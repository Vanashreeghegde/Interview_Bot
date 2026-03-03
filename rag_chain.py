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

# ==================================================
# GET NEXT QUESTION (Phase + Adaptive Difficulty + Feedback)
# ==================================================

def get_next_question(topic, difficulty, history, used_questions, penalized_questions=None):
    """
    history: List of dictionaries with previous scores
    used_questions: List of strings of questions already asked
    penalized_questions: List of questions the user marked as 'useless'
    """
    question_count = len(history)
    penalized_questions = penalized_questions or []

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
    # PREPARE HISTORY FOR PROMPT
    # ---------------------------
    history_str = "\n- ".join(used_questions[-5:]) if used_questions else "None"
    penalties_str = "\n- ".join(penalized_questions[-3:]) if penalized_questions else "None"

    # ---------------------------
    # PHASE-BASED QUERY
    # ---------------------------
    if phase == 1:
        query = f"{topic} python list tuple dict definition basics interview questions difficulty {difficulty}"
        instruction = "Ask a crisp conceptual question about Python fundamentals or ML basics."
    elif phase == 2:
        query = f"{topic} real world case study production tradeoffs business problem difficulty {difficulty}"
        instruction = "Ask a realistic ML scenario or case study question."
    else:
        query = "Python SQL algorithmic coding logic challenges hacker rank style"
        instruction = "Ask a coding challenge. Provide a problem statement (HackerRank style)."

    docs = retriever.retrieve(query)

    if not docs:
        context = f"Standard Data Science concepts related to {topic}."
    else:
        context = "\n\n".join([doc.page_content[:800] for doc in docs[:3]])

    # ---------------------------
    # STRICT GROUNDED PROMPT (With Learning from Penalties)
    # ---------------------------
    prompt = f"""
You are a Professional Data Science Interviewer. 

PREVIOUSLY ASKED (DO NOT REPEAT):
- {history_str}

USER FEEDBACK (THESE WERE LABELED 'USELESS' - AVOID THESE TYPES):
- {penalties_str}

Phase: {phase}
Difficulty Level: {difficulty}
Instruction: {instruction}

Rules:
1. DO NOT ask the "primary goal of a data science interview."
2. DO NOT ask the same concept in different ways.
3. If Phase 3: Present a coding problem with clear input/output expectations.
4. Focus on technical depth (Feature Engineering, Evaluation, Algorithms).
5. Use ONLY information from CONTEXT.
6. Must end with ONE question mark. No intro, no explanation.

CONTEXT:
{context}

QUESTION:"""

    question = generate_response(
        prompt,
        model=GENERATION_MODEL,
        temperature=0.6, # Slightly higher for more variety
        max_tokens=300
    ).strip()

    # Ensure it's a question
    if not question.endswith("?"):
        question += "?"

    # ---------------------------
    # FINAL REPETITION CHECK (String Matching)
    # ---------------------------
    for prev in used_questions:
        # Check if the first 40 chars match any previous question to catch rephrasing
        if question.lower()[:40] in prev.lower():
            # Fallback if AI tries to repeat
            return f"Discuss the tradeoffs of using different evaluation metrics for {topic}?", difficulty

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
