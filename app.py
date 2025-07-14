import streamlit as st
import os
import openai
import groq
import dotenv

dotenv.loadenv()


# Set up the env variables
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

