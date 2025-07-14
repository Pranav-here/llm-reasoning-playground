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

st.set_page_config(page_title="Chain of Thought Demo", layout="centered")
st.title("Chain of Thought vs Direct Prompting")

# initialize session state for Tree-of-Thought
if "tree_thoughts" not in st.session_state:
    st.session_state.tree_thoughts = []

test_suite = {
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

selected_test = st.selectbox(
    "Pick a reasoning test (or choose None):",
    list(test_suite.keys())
)
if selected_test != "None (custom question)":
    question = test_suite[selected_test]
else:
    question = ""

question = st.text_area(
    "Enter a question:",
    value=question,
    placeholder="What's 12 times 17?"
)
model_choice = st.selectbox(
    "Choose model:",
    ["OpenAI (GPT 3.5)", "Groq (gemma2-9b-it)"]
)
mode = st.radio(
    "Choose prompting style:",
    ["Direct", "Chain of Thought"]
)
use_self_consistency = st.checkbox("Enable Self-Consistency (n=5)")
use_tree_of_thought = st.checkbox("Enable Tree-of-Thought Mode (3 reasoning paths)")
use_reflexion = st.checkbox("Enable Reflexion Agent Mode")

def build_prompt(q, mode):
    return f"{q.strip()}\n\nLet's think step by step." if mode == "Chain of Thought" else q.strip()

def call_model(prompt, model_choice):
    if model_choice == "OpenAI (GPT 3.5)":
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

def get_self_consistent_answer(prompt, model_choice, n=5):
    answers = []
    for _ in range(n):
        ans, _ = call_model(prompt, model_choice)
        answers.append(ans.strip().split("\n")[-1])
    most_common = Counter(answers).most_common(1)[0][0]
    return most_common, answers

def get_tree_of_thought(prompt, model_choice, n=3):
    return [call_model(prompt, model_choice)[0].strip() for _ in range(n)]

def vote_best_path_with_llm(paths, original_prompt, model_choice):
    voting_prompt = f"""You are given three different reasoning paths answering the following question:

{original_prompt}

Reasoning Path 1:
{paths[0]}

Reasoning Path 2:
{paths[1]}

Reasoning Path 3:
{paths[2]}

Analyze the logic and correctness of each answer. Then pick the best one and explain why it is better than the others.

Reply in this format:

Best Path: 1/2/3  
Justification: <your reasoning>
"""
    vote_response, _ = call_model(voting_prompt, model_choice)

    # Extract which path it chose
    best_path_num = 1  # default fallback
    for i in [1, 2, 3]:
        if f"Best Path: {i}" in vote_response:
            best_path_num = i
            break

    return best_path_num, vote_response.strip()

def run_reflexion_loop(prompt, model_choice, max_retries=1):
    original_answer, _ = call_model(prompt, model_choice)

    # Ask the model to critique its own answer
    critique_prompt = f"""
You previously answered the following question:

{prompt}

Your answer was:

\"\"\"{original_answer}\"\"\"

Your task now is to critique this answer. If it is correct, explain why. If there are any reasoning errors, incorrect calculations, or missing steps, clearly point them out. Be objective.
"""

    reflection, _ = call_model(critique_prompt, model_choice)

    # Inject feedback into a new prompt if retrying
    retry_prompt = f"""
You previously answered the question but made some errors or were unsure.

Here is a reflection of your last attempt:

{reflection}

Now try again and provide an improved, step-by-step answer:
{prompt}
"""
    improved_answer, _ = call_model(retry_prompt, model_choice)

    return original_answer.strip(), reflection.strip(), improved_answer.strip()


# generate
if st.button("Generate Answer"):
    with st.spinner("Thinking..."):
        prompt = build_prompt(question, mode)

        if use_self_consistency and not use_tree_of_thought and not use_reflexion:
            final, all_answers = get_self_consistent_answer(prompt, model_choice)
            st.markdown("### Self-Consistent Answer")
            st.write(final)
            with st.expander("See All Generated Answers"):
                for i, ans in enumerate(all_answers, 1):
                    st.markdown(f"**Try {i}:** {ans}")

        elif use_tree_of_thought:
            prompt = build_prompt(question, "Chain of Thought")
            st.session_state.tree_thoughts = get_tree_of_thought(prompt, model_choice)

            st.markdown("### 🌳 Tree of Thought Responses")
            all_thoughts = get_tree_of_thought(prompt, model_choice)

            # LLM voting
            best_path_num, justification = vote_best_path_with_llm(all_thoughts, question, model_choice)

            st.markdown(f"**🧠 LLM Selected Best Path: Path {best_path_num}**")
            st.markdown("**🗣️ Justification:**")
            st.info(justification)

            with st.expander("🌿 All Reasoning Paths"):
                for i, thought in enumerate(all_thoughts, 1):
                    st.markdown(f"**Path {i}:**\n\n{thought}")


        elif use_reflexion:
            prompt = build_prompt(question, "Chain of Thought")
            original, reflection, improved = run_reflexion_loop(prompt, model_choice)

            st.markdown("### 🪞 Reflexion Agent Output")
            st.markdown("**First Attempt:**")
            st.write(original)

            st.markdown("**🔍 Self-Critique / Reflection:**")
            st.info(reflection)

            st.markdown("**🔁 Improved Answer After Reflection:**")
            st.success(improved)


        else:
            answer, tokens_used = call_model(prompt, model_choice)
            st.markdown("### Answer")
            st.write(answer)
            if tokens_used:
                cost = tokens_used / 1000 * 0.001
                st.info(f"Used {tokens_used} tokens — estimated cost: ${cost:.4f}")

# Tree-of-Thought UI (persistent)
if use_tree_of_thought and st.session_state.tree_thoughts:
    st.markdown("### Tree of Thought Responses")
    selection = st.radio(
        "Pick the best response:",
        [f"Path{i+1}" for i in range(len(st.session_state.tree_thoughts))],
        key="thought_selection"
    )
    idx = int(selection.replace("Path", "")) - 1
    st.markdown("**Selected Reasoning Path:**")
    st.write(st.session_state.tree_thoughts[idx])
    with st.expander("See All Paths"):
        for i, th in enumerate(st.session_state.tree_thoughts, 1):
            st.markdown(f"**Path {i}:**\n\n{th}")
