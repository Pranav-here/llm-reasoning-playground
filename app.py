import streamlit as st
import os
import openai
import groq
import dotenv

dotenv.load_dotenv()


# Set up the env variables
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Chain of Thought Demo", layout='centered')
st.title('Chain of Thought vs Direct Prompting')

question=st.text_area("Enter a question:", placeholder="Whats 12 times 17?")

model_choice=st.selectbox("Choose model:", ["OpenAI (GPT 3.5)", "Groq (Mistral)"])
mode=st.radio("Choose prompting style:", ["Direct", "Chain of Thought"])


