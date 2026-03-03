import streamlit as st
import time
import random
from streamlit_autorefresh import st_autorefresh
from rag_chain import get_next_question, evaluate_answer, generate_final_feedback

# ===============================
# CONFIG
# ===============================
TOTAL_SECONDS = 45 * 60  # 45 minutes
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
    "Generative AI",
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
    "used_questions": [],
    "topic_scores": {}  # Track scores per topic for adaptive topic selection
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ===============================
# HELPER FUNCTION
# ===============================
def pick_topic(topics, topic_scores):
    """
    Pick topic with least coverage (adaptive), else random.
    """
    unasked = [t for t in topics if t not in topic_scores or len(topic_scores[t]) == 0]
    return random.choice(unasked) if unasked else random.choice(topics)

# ===============================
# SIDEBAR TIMER
# ===============================
with st.sidebar:
    st.header("⏳ Timer")
    if st.session_state.started and not st.session_state.interview_complete:
        if st.session_state.start_time is None:
            st.session_state.start_time = time.time()

        st_autorefresh(interval=1000, key="timer")

        elapsed = time.time() - st.session_state.start_time
        remaining = max(0, TOTAL_SECONDS - elapsed)
        mins = int(remaining // 60)
        secs = int(remaining % 60)
        st.metric("Time Left", f"{mins:02d}:{secs:02d}")

        if remaining <= 0:
            st.session_state.interview_complete = True
            st.rerun()

        if st.button("End Interview"):
            st.session_state.interview_complete = True
            st.rerun()

# ===============================
# START SCREEN
# ===============================
if not st.session_state.started:
    st.title("🕵🏻‍♂️ Personalized DS Interview Bot 🤖")
    st.markdown("Ready to dive in and test your knowledge 🤗")
    if st.button("Start Interview"):
        st.session_state.started = True
        st.session_state.start_time = time.time()

        topic = pick_topic(TOPICS, st.session_state.topic_scores)
        st.session_state.current_topic = topic

        question, difficulty = get_next_question(
            topic=topic,
            difficulty=st.session_state.difficulty,
            history=[],
            used_questions=[]
        )

        st.session_state.current_question = question
        st.session_state.difficulty = difficulty
        st.session_state.used_questions.append(question)

# ===============================
# INTERVIEW LOOP
# ===============================
elif st.session_state.started and not st.session_state.interview_complete:

    st.write(f"**Topic:** {st.session_state.current_topic} | Difficulty: {st.session_state.difficulty}")
    st.info(st.session_state.current_question)

    question_index = len(st.session_state.history)

    user_answer = st.text_area(
        "Your Answer:",
        height=150,
        key=f"answer_{question_index}"
    )

    col1, col2 = st.columns([1,1])

    # -------------------
    # Submit Answer
    # -------------------
    with col1:
        if st.button("Submit Answer", key=f"submit_{question_index}"):
            if not user_answer.strip():
                st.warning("Please enter an answer before submitting.")
            else:
                with st.spinner("Evaluating..."):
                    score = evaluate_answer(
                        question=st.session_state.current_question,
                        answer=user_answer
                    )

                st.session_state.scores.append(score)
                st.session_state.history.append({
                    "topic": st.session_state.current_topic,
                    "question": st.session_state.current_question,
                    "answer": user_answer,
                    "score": score
                })

                topic = st.session_state.current_topic
                if topic not in st.session_state.topic_scores:
                    st.session_state.topic_scores[topic] = []
                st.session_state.topic_scores[topic].append(score)

                # -------------------
                # Automatically fetch next question
                # -------------------
                next_topic = pick_topic(TOPICS, st.session_state.topic_scores)
                st.session_state.current_topic = next_topic

                next_question, difficulty = get_next_question(
                    topic=next_topic,
                    difficulty=st.session_state.difficulty,
                    history=st.session_state.history,
                    used_questions=st.session_state.used_questions
                )

                st.session_state.current_question = next_question
                st.session_state.difficulty = difficulty
                st.session_state.used_questions.append(next_question)

                st.rerun()  # go to next question immediately

    # -------------------
    # Skip Question
    # -------------------
    with col2:
        if st.button("Skip Question", key=f"skip_{question_index}"):
            next_topic = pick_topic(TOPICS, st.session_state.topic_scores)
            st.session_state.current_topic = next_topic

            skipped_question, difficulty = get_next_question(
                topic=next_topic,
                difficulty=st.session_state.difficulty,
                history=st.session_state.history,
                used_questions=st.session_state.used_questions
            )

            st.session_state.current_question = skipped_question
            st.session_state.difficulty = difficulty
            st.session_state.used_questions.append(skipped_question)

            st.info("Question skipped. Here’s your next question.")
            st.rerun()

# ===============================
# INTERVIEW RESULTS
# ===============================
elif st.session_state.interview_complete:
    st.title("Interview Finished")
    st.markdown("Hureyy 🎉🎉 you did it!!")

    if st.session_state.history:
        with st.spinner("Generating final feedback..."):
            feedback = generate_final_feedback(
                history=st.session_state.history
            )
        st.markdown(feedback)
    else:
        st.warning("No answers were submitted.")

    if st.button("Restart Interview"):
        for key in list(defaults.keys()):
            if key in st.session_state:
                del st.session_state[key]