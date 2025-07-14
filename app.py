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

question = st.text_area("Enter a question:", placeholder="What's 12 times 17?")
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
