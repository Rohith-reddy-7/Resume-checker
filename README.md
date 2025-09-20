# Automated Resume Relevance Checker

## Project Overview
This project evaluates how well a candidate’s resume matches a job description. It has two modes:

- **Simple Keyword Match** – Extracts keywords from resume and JD, calculates a relevance score, and provides suggestions.
- **LLM-powered Mode** – Uses OpenAI GPT models to provide structured analysis of skills, education, experience, and overall relevance.

Users can upload resumes (PDF, DOCX, TXT), paste a job description, and see a relevance score along with matched skills and improvement suggestions.

---

## How to Run Locally

1. **Clone the repository**
```bash
git clone https://github.com/Rohith-reddy-7/Resume-checker
cd Resume-checker
## How to Run Locally

### 1. Install dependencies
```bash
pip install -r requirements.txt

##Windows##

setx OPENAI_API_KEY "your-api-key"

##Features##

Relevance score (0–100)

Matched skills from resume

Job description keywords extraction

Suggestions to improve resume relevance

Works without API if OpenAI quota is exceeded (Simple mode fallback)

##Notes##

LLM-powered mode uses OpenAI GPT models and requires a valid API key.

OpenAI free trial credits may run out; in that case, the Simple mode ensures full functionality.

**Links**

GitHub Repository: https://github.com/Rohith-reddy-7/Resume-checker

Deployed Web App: [https://resume-checker-8zlv9scgnsohrwsqkammpr.streamlit.app/]

Demo Video: [[YouTube Link]--(https://www.youtube.com/watch?v=Vn_hfGyYlc0)]

