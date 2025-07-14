import streamlit as st
import os
import openai
from groq import Groq
import dotenv
from collections import Counter

dotenv.load_dotenv()

# Set up the env variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Chain of Thought Demo", layout='centered')
st.title('Chain of Thought vs Direct Prompting')

test_suite = {
    "None (custom question)": "",

    # --- Math & Quantitative Reasoning ---
    "Olympiad Algebra": (
        "Let f(x) = x² + 3x + 1.  Find all real x such that "
        "f(f(f(x))) = 1."
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
        "A fair coin is tossed until either two consecutive heads appear *or* "
        "three total tails appear—whichever happens first.  What is the probability "
        "that the process ends with two consecutive heads?"
    ),

    # --- Multi-Step Logic & Deduction ---
    "Knights-Knaves-Day": (
        "On an island of knights (always truthful) and knaves (always lie), "
        "three inhabitants — Alice, Bob, and Carol — make these statements:\n"
        "• Alice: “Exactly one of us is a knight.”\n"
        "• Bob: “Carol and I are of the same type.”\n"
        "• Carol: “Alice is a knight or Bob is a knave.”\n"
        "Determine who’s a knight and who’s a knave."
    ),
    "Temporal Logic Paradox": (
        "A professor tells their class: “On exactly one of the next five weekdays "
        "I will give a surprise quiz.  You won’t know the quiz day until the morning "
        "of the quiz.”  Using backward-induction reasoning, the students conclude "
        "a surprise quiz is impossible—yet the professor gives it on Wednesday and "
        "everyone is surprised.  Identify the flawed logical step in the students’ "
        "reasoning *and* formally state why the quiz could still be a surprise."
    ),

    # --- Adversarial Riddles & Lateral Thinking ---
    "Counterfactual Riddle": (
        "You are told that if statement X is true then statement Y is false, "
        "and if X is false then Y is true.  You also learn that either X or Y "
        "is definitely true (but not both).  Without knowing the actual truth of X, "
        "determine the truth values of X and Y and justify your answer."
    ),
    "Impossible Object Reasoning": (
        "Imagine a two-dimensional staircase drawn in an impossible loop (a Penrose "
        "staircase).  If you start at the bottom step and ascend one full loop of "
        "the staircase, do you end up higher than where you started?  Give a rigorous "
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


selected_test=st.selectbox("Pick a reasoning test( or choose None):", list(test_suite.keys()))
if selected_test != "None (custom question)":
    question=test_suite[selected_test]
else:
    question=""

question = st.text_area("Enter a question:",value=question, placeholder="What's 12 times 17?")
model_choice = st.selectbox("Choose model:", ["OpenAI (GPT 3.5)", "Groq (gemma2-9b-it)"])
mode = st.radio("Choose prompting style:", ["Direct", "Chain of Thought"])
use_self_consistency = st.checkbox("Enable Self-Consistency (n=5)")

def build_prompt(q, mode):
    return f"{q.strip()}\n\nLet's think step by step." if mode == "Chain of Thought" else q.strip()

# Model call logic
def call_model(prompt, model_choice):
    if model_choice == "OpenAI (GPT 3.5)":
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content, response.usage.total_tokens

    elif model_choice == "Groq (gemma2-9b-it)":
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="gemma-9b-it",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content, None  # Groq SDK doesn’t give usage yet

def get_self_consistent_answer(prompt, model_choice, n=5):
    answers = []
    for _ in range(n):
        ans, _ = call_model(prompt, model_choice)
        final_line = ans.strip().split("\n")[-1]
        answers.append(final_line)
    count = Counter(answers)
    most_common = count.most_common(1)[0][0]
    return most_common, answers

# Run
if st.button("Generate Answer"):
    with st.spinner("Thinking..."):
        prompt = build_prompt(question, mode)

        if use_self_consistency:
            final, all_answers = get_self_consistent_answer(prompt, model_choice)
            st.markdown("### Self-Consistent Answer")
            st.write(final)
            with st.expander("See All Generated Answers"):
                for i, ans in enumerate(all_answers, 1):
                    st.markdown(f"**Try {i}:** {ans}")
        else:
            answer, tokens_used = call_model(prompt, model_choice)
            st.markdown("### Answer")
            st.write(answer)

            if tokens_used:
                cost = tokens_used / 1000 * 0.001
                st.info(f"Used {tokens_used} tokens — estimated cost: ${cost:.4f}")
