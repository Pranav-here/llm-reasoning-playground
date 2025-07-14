import streamlit as st
import os
import openai
from groq import Groq
import dotenv

dotenv.load_dotenv()


# Set up the env variables
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Chain of Thought Demo", layout='centered')
st.title('Chain of Thought vs Direct Prompting')

question=st.text_area("Enter a question:", placeholder="Whats 12 times 17?")

model_choice=st.selectbox("Choose model:", ["OpenAI (GPT 3.5)", "Groq (gemma2-9b-it)"])
mode=st.radio("Choose prompting style:", ["Direct", "Chain of Thought"])

if st.button("Generate Answer"):
    with st.spinner("Thinking..."):
        if mode == 'Chain of Thought':
            prompt=f'{question}\n\n Lets think step by step'
        else:
            prompt=question

    if model_choice == "OpenAI (GPT 3.5)":
        client=openai.OpenAI(api_key=OPENAI_API_KEY)

        response=client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        answer=response.choices[0].message.content

    elif model_choice == "Groq (gemma2-9b-it)":
            client = Groq(api_key=GROQ_API_KEY)

            response = client.chat.completions.create(
                model="gemma2-9b-it",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            answer = response.choices[0].message.content

    st.markdown('### Answer')
    st.write(answer)


