# LLM Reasoning Playground

## Overview
LLM Reasoning Playground is a Streamlit web application that lets you benchmark large‑language‑model reasoning strategies in real time.  
The app supports:

* Direct prompting
* Chain‑of‑Thought (CoT)
* Self‑Consistency voting
* Tree‑of‑Thought (ToT) path generation with automatic model voting
* Reflexion loops with self‑critique and retry

It works with both OpenAI GPT‑3.5 and Groq Gemma‑9B‑IT models.

## Features
| Module            | Purpose                                                            |
|-------------------|--------------------------------------------------------------------|
| Direct / CoT      | Compare terse answers with step‑by‑step reasoning                 |
| Self‑Consistency  | Generate *n* CoT paths and return the majority answer              |
| Tree‑of‑Thought   | Create three independent reasoning paths and let the LLM rank them |
| Reflexion Agent   | Critique its own answer, then retry with injected feedback         |

## Quick Start (Local)
```bash
git clone https://github.com/pranav-here/llm‑reasoning‑playground.git
cd llm‑reasoning‑playground
cp .env.example .env            # add your API keys
pip install -r requirements.txt
streamlit run app.py
```

## Quick Start (Docker)
```bash
docker build -t llm-reasoning-playground .
docker run -p 8501:8501 --env-file .env llm-reasoning-playground
```

## Environment Variables
```
OPENAI_API_KEY=<your-openai-key>
GROQ_API_KEY=<your-groq-key>
```

## Project Structure
```
├── app.py            # Streamlit application
├── requirements.txt
├── Dockerfile
└── README.md
```

## License
MIT
