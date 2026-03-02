import streamlit as st
import time
import random
from streamlit_autorefresh import st_autorefresh
from rag_chain import get_next_question, evaluate_answer, generate_final_feedback

# ===============================
# CONFIG
# ===============================
TOTAL_SECONDS = 45 * 60  # 45 min
TOPICS = [
    "Machine Learning",
    "Statistics",
    "SQL",
    "Python",
    "Deep Learning",
    "Model Evaluation",
    "Feature Engineering",
    "Data Cleaning",
    "ML System Design",
    "Genarative AI",
    "LLM",
    "RAG"
]

st.set_page_config(page_title="DS Interview Bot", layout="wide")

# ===============================
# SESSION STATE INIT
# ===============================
defaults = {
    "started": False,
    "start_time": None,
    "current_question": None,
    "current_topic": None,
    "difficulty": 1,
    "interview_complete": False,
    "history": [],
    "scores": [],
    "used_questions": []
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ===============================
# SIDEBAR TIMER
# ===============================
with st.sidebar:
    st.header("⏳ Timer")
    if st.session_state.started and not st.session_state.interview_complete:
        # Autorefresh every 1 second
        st_autorefresh(interval=1000, key="timer")
        elapsed = int(time.time() - st.session_state.start_time)
        remaining = max(0, TOTAL_SECONDS - elapsed)
        mins, secs = remaining // 60, remaining % 60
        st.metric("Time Left", f"{mins:02d}:{secs:02d}")

        # Auto end interview if timer runs out
        if remaining <= 0:
            st.session_state.interview_complete = True

        if st.button("End Interview"):
            st.session_state.interview_complete = True

# ===============================
# START SCREEN
# ===============================
if not st.session_state.started:
    st.title("Data Science Technical Interview")
    if st.button("Start Interview"):
        st.session_state.started = True
        st.session_state.start_time = time.time()
        # initial question
        topic = random.choice(TOPICS)
        st.session_state.current_topic = topic
        question = get_next_question(
            topic=topic,
            difficulty=st.session_state.difficulty,
            used_questions=[]
        )
        st.session_state.current_question = question
        st.session_state.used_questions.append(question)

# ===============================
# INTERVIEW LOOP
# ===============================
elif st.session_state.started and not st.session_state.interview_complete:
    st.write(f"**Topic:** {st.session_state.current_topic} | Difficulty: {st.session_state.difficulty}")
    st.info(st.session_state.current_question)

    user_answer = st.text_area(
        "Your Answer:",
        height=150,
        key=f"answer_{len(st.session_state.history)}"
    )
    submitted = st.button("Submit Answer", key=f"submit_{len(st.session_state.history)}")

    if submitted:
        if not user_answer.strip():
            st.warning("Please enter an answer before submitting.")
        else:
            # Evaluate answer
            with st.spinner("Evaluating..."):
                score = evaluate_answer(
                    question=st.session_state.current_question,
                    answer=user_answer
                )

            # Update session state
            st.session_state.scores.append(score)
            st.session_state.history.append({
                "topic": st.session_state.current_topic,
                "question": st.session_state.current_question,
                "answer": user_answer,
                "score": score
            })

            # Adaptive difficulty
            if score >= 7:
                st.session_state.difficulty += 1
            elif score <= 4:
                st.session_state.difficulty = max(1, st.session_state.difficulty - 1)

            # Load next question
            topic = random.choice(TOPICS)
            st.session_state.current_topic = topic
            next_question = get_next_question(
                topic=topic,
                difficulty=st.session_state.difficulty,
                used_questions=st.session_state.used_questions
            )
            st.session_state.current_question = next_question
            st.session_state.used_questions.append(next_question)

# ===============================
# INTERVIEW RESULTS
# ===============================
elif st.session_state.interview_complete:
    st.title("Interview Finished")

    if st.session_state.history:
        average_score = round(sum(st.session_state.scores) / len(st.session_state.scores), 2)
        with st.spinner("Generating final feedback..."):
            feedback = generate_final_feedback(
                history=st.session_state.history,
                average_score=average_score,
                final_difficulty=st.session_state.difficulty
            )
        st.markdown(feedback)
    else:
        st.warning("No answers were submitted.")

    if st.button("Restart Interview"):
        for key in list(defaults.keys()):
            if key in st.session_state:
                del st.session_state[key]