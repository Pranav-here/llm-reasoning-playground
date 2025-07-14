import streamlit as st
import os
import openai
from groq import Groq
import dotenv
from collections import Counter

# -------------------------------------------------------------
# 🔧 Environment & Configuration
# -------------------------------------------------------------
dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

APP_TITLE = "LLM Reasoning Playground"

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# -------------------------------------------------------------
# 📚 Advanced Reasoning Test‑Suite (unchanged)
# -------------------------------------------------------------
TEST_SUITE = {
    "None (custom question)": "",
    # --- Math & Quantitative Reasoning ---
    "Olympiad Algebra": (
        "Let f(x) = x² + 3x + 1.  Find all real x such that f(f(f(x))) = 1."
    ),
    "Number-Theory Puzzle": (
        "Find the smallest positive integer N such that N is divisible by 2025, "
        "its digit sum equals 18, and reversing its digits yields a prime number."
    ),
    "Geometry (Hidden Constraint)": (
        "In triangle ABC, AB = 13, AC = 15, and the altitude from A meets BC at D "
        "with AD = 12.  Find the length of BC."
    ),
    "Probability (Nested Events)": (
        "A fair coin is tossed until either two consecutive heads appear *or* three "
        "total tails appear—whichever happens first.  What is the probability that "
        "the process ends with two consecutive heads?"
    ),
    # --- Multi-Step Logic & Deduction ---
    "Knights-Knaves-Day": (
        "On an island of knights (always truthful) and knaves (always lie), three "
        "inhabitants — Alice, Bob, and Carol — make these statements:\n"
        "• Alice: “Exactly one of us is a knight.”\n"
        "• Bob: “Carol and I are of the same type.”\n"
        "• Carol: “Alice is a knight or Bob is a knave.”\n"
        "Determine who’s a knight and who’s a knave."
    ),
    "Temporal Logic Paradox": (
        "A professor tells their class: “On exactly one of the next five weekdays "
        "I will give a surprise quiz.  You won’t know the quiz day until the morning "
        "of the quiz.”  Using backward-induction reasoning, the students conclude a "
        "surprise quiz is impossible—yet the professor gives it on Wednesday and "
        "everyone is surprised.  Identify the flawed logical step in the students’ "
        "reasoning *and* formally state why the quiz could still be a surprise."
    ),
    # --- Adversarial Riddles & Lateral Thinking ---
    "Counterfactual Riddle": (
        "You are told that if statement X is true then statement Y is false, and if "
        "X is false then Y is true.  You also learn that either X or Y is definitely "
        "true (but not both).  Without knowing the actual truth of X, determine the "
        "truth values of X and Y and justify your answer."
    ),
    "Impossible Object Reasoning": (
        "Imagine a two-dimensional staircase drawn in an impossible loop (a Penrose "
        "staircase).  If you start at the bottom step and ascend one full loop of the "
        "staircase, do you end up higher than where you started?  Give a rigorous "
        "explanation using graph theory or topology rather than visual intuition."
    ),
    # --- Complex Word Problems ---
    "Work-Rate with Leaks": (
        "Tank A and Tank B are identical and initially empty.  Pipe P can fill a tank "
        "in 3 hours, Pipe Q in 4 hours, and Pipe R in 6 hours.  Pipe R is also a leak "
        "when run in reverse, emptying 5 liters per minute.  All three pipes are "
        "opened on Tank A for 30 minutes, then R is reversed (becoming a leak) while "
        "P and Q continue.  Exactly when (HH:MM) will Tank A be full?"
    ),
    "Multi-Stage Exchange": (
        "Alex, Blair, and Casey start with $x,$ $y,$ and $z$ dollars respectively.  "
        "Alex gives Blair half of what Alex has.  Blair then gives Casey a third of "
        "what Blair now has.  Casey finally gives Alex a quarter of Casey’s balance.  "
        "After these exchanges everyone ends with $20.  Find the original values of "
        "$x, y,$ and $z.$"
    ),
}

# Persistent state for Tree‑of‑Thought paths
if "tree_thoughts" not in st.session_state:
    st.session_state.tree_thoughts = []

# -------------------------------------------------------------
# 🖥️ Sidebar – Controls & Settings
# -------------------------------------------------------------
st.sidebar.header("⚙️ Controls")
selected_test = st.sidebar.selectbox("Reasoning Test", list(TEST_SUITE.keys()), index=0)

question_default = TEST_SUITE[selected_test]
question = st.text_area("✍️ Prompt", value=question_default, height=120, placeholder="Enter your own question…")

model_choice = st.sidebar.selectbox("Model", ["OpenAI (GPT 3.5)", "Groq (gemma2-9b-it)"])
mode = st.sidebar.radio("Prompting Style", ["Direct", "Chain of Thought"], index=1)

st.sidebar.markdown("---")
use_self_consistency = st.sidebar.checkbox("🔁 Self‑Consistency (n=5)")
use_tree_of_thought = st.sidebar.checkbox("🌳 Tree‑of‑Thought (3 paths)")
use_reflexion = st.sidebar.checkbox("🪞 Reflexion Agent")

# Ensure only one advanced mode is active
advanced_modes = sum([use_self_consistency, use_tree_of_thought, use_reflexion])
if advanced_modes > 1:
    st.sidebar.error("Select **only one** advanced mode at a time.")

# -------------------------------------------------------------
# 🔌 Model Helpers (logic unchanged)
# -------------------------------------------------------------

def build_prompt(q: str, prompt_mode: str) -> str:
    """Attach CoT cue if needed."""
    return f"{q.strip()}\n\nLet's think step by step." if prompt_mode == "Chain of Thought" else q.strip()


def call_model(prompt: str, choice: str):
    if choice == "OpenAI (GPT 3.5)":
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return resp.choices[0].message.content, resp.usage.total_tokens

    client = Groq(api_key=GROQ_API_KEY)
    resp = client.chat.completions.create(
        model="gemma-9b-it",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return resp.choices[0].message.content, None


# ---------- Self‑Consistency ----------

def self_consistent_answer(prompt: str, choice: str, n: int = 5):
    answers = [call_model(prompt, choice)[0].split("\n")[-1].strip() for _ in range(n)]
    most_common = Counter(answers).most_common(1)[0][0]
    return most_common, answers


# ---------- Tree‑of‑Thought ----------

def tree_of_thought(prompt: str, choice: str, paths: int = 3):
    return [call_model(prompt, choice)[0].strip() for _ in range(paths)]


def vote_best_path(paths, original_prompt, choice):
    voting_prompt = f"""You are given three different reasoning paths answering the following question:\n\n{original_prompt}\n\nReasoning Path 1:\n{paths[0]}\n\nReasoning Path 2:\n{paths[1]}\n\nReasoning Path 3:\n{paths[2]}\n\nAnalyze the logic and correctness of each answer. Then pick the best one and explain why it is better than the others.\n\nReply exactly in this format:\nBest Path: 1/2/3\nJustification: <your reasoning>"""
    response, _ = call_model(voting_prompt, choice)
    best_num = next((i for i in [1, 2, 3] if f"Best Path: {i}" in response), 1)
    return best_num, response.strip()


# ---------- Reflexion ----------

def reflexion_loop(prompt: str, choice: str):
    first_answer, _ = call_model(prompt, choice)
    critique_prompt = f"""You previously answered:\n\n---\n{first_answer}\n---\n\nCritique this answer objectively. If it is fully correct, say so; otherwise point out errors and suggest fixes."""
    critique, _ = call_model(critique_prompt, choice)

    retry_prompt = f"""Using the feedback below, improve your answer step‑by‑step:\n\n{critique}\n\nQuestion: {prompt}"""
    improved, _ = call_model(retry_prompt, choice)
    return first_answer.strip(), critique.strip(), improved.strip()

# -------------------------------------------------------------
# 🚀 Main Run Button
# -------------------------------------------------------------
if st.button("Generate", type="primary") and question.strip():
    with st.spinner("Thinking..."):
        final_prompt = build_prompt(question, mode)

        # ----- Self‑Consistency -----
        if use_self_consistency and advanced_modes == 1:
            best, tries = self_consistent_answer(final_prompt, model_choice)
            st.subheader("🔁 Self‑Consistent Answer")
            st.write(best)
            with st.expander("See all tries"):
                for i, ans in enumerate(tries, 1):
                    st.markdown(f"**Try {i}:** {ans}")

        # ----- Tree‑of‑Thought -----
        elif use_tree_of_thought and advanced_modes == 1:
            final_prompt = build_prompt(question, "Chain of Thought")  # always CoT
            paths = tree_of_thought(final_prompt, model_choice)
            best_num, justification = vote_best_path(paths, question, model_choice)

            st.subheader(f"🌳 LLM‑Chosen Best Path: Path {best_num}")
            st.info(justification)

            with st.expander("Show all reasoning paths"):
                for i, p in enumerate(paths, 1):
                    st.markdown(f"**Path {i}:**\n\n{p}")

        # ----- Reflexion -----
        elif use_reflexion and advanced_modes == 1:
            final_prompt = build_prompt(question, "Chain of Thought")
            original, critique, improved = reflexion_loop(final_prompt, model_choice)
            st.subheader("🪞 Reflexion Agent")
            st.markdown("**First Attempt:**")
            st.write(original)
            st.markdown("**Critique:**")
            st.info(critique)
            st.markdown("**Improved Answer:**")
            st.success(improved)

        # ----- Basic Direct / CoT -----
        elif advanced_modes == 0:
            answer, tokens = call_model(final_prompt, model_choice)
            st.subheader("💡 Answer")
            st.write(answer)
            if tokens:
                cost = tokens / 1000 * 0.001
                st.caption(f"Tokens: {tokens} | Estimated cost: ${cost:.4f}")

        else:
            st.warning("Please choose **only one** advanced mode at a time.")
