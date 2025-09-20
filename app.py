# app.py
import streamlit as st
import io, os, re, json
from collections import Counter

# Optional libraries
try:
    import pdfplumber
except Exception:
    pdfplumber = None
try:
    from docx import Document
except Exception:
    Document = None

# OpenAI import for LLM mode
try:
    import openai
except Exception:
    openai = None

st.set_page_config(page_title="Resume Relevance Checker", layout="wide")

###########################
# Utility: text extraction
###########################
def extract_text_from_file(uploaded_file):
    if uploaded_file is None:
        return ""
    fname = uploaded_file.name.lower()
    content = ""
    data = uploaded_file.read()
    if fname.endswith(".pdf"):
        if not pdfplumber:
            st.error("pdfplumber not installed. pip install pdfplumber")
            return ""
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            pages = []
            for p in pdf.pages:
                txt = p.extract_text()
                if txt:
                    pages.append(txt)
            content = "\n".join(pages)
    elif fname.endswith(".docx") or fname.endswith(".doc"):
        if not Document:
            st.error("python-docx not installed. pip install python-docx")
            return ""
        doc = Document(io.BytesIO(data))
        content = "\n".join([p.text for p in doc.paragraphs])
    else:
        try:
            content = data.decode('utf-8')
        except Exception:
            content = str(data)
    return content

###########################
# Simple keyword extraction
###########################
STOPWORDS = set([
    "and","the","for","with","that","this","from","will","have","has","a","an","in","on","to","of","is","are","be",
    "by","as","or","we","our","you","your","at","it","they","their","skill","skills"
])

def extract_keywords_basic(text, top_k=40):
    tokens = re.findall(r"[A-Za-z#+\-\.\d]+", text)
    tokens = [t.lower() for t in tokens if len(t)>1]
    tokens = [t for t in tokens if t not in STOPWORDS and not t.isdigit()]
    counts = Counter(tokens)
    most = [w for w,_ in counts.most_common(top_k)]
    return most

###########################
# Simple scoring algorithm
###########################
def simple_score(resume_text, jd_text):
    resume_tokens = set(re.findall(r"[A-Za-z#+\-\.\d]+", resume_text.lower()))
    jd_keywords = extract_keywords_basic(jd_text, top_k=50)
    if len(jd_keywords)==0:
        jd_keywords = extract_keywords_basic(jd_text + " skills", top_k=10)
    jd_set = set(jd_keywords)
    if not jd_set:
        return {"score": 0, "match_pct": 0, "matched": [], "jd_keywords": jd_keywords, "details": {}}
    matched = sorted(list(jd_set.intersection(resume_tokens)))
    match_pct = len(matched) / max(1, len(jd_set))
    # education match (basic)
    edu_found = any(k in resume_text.lower() for k in ["btech","bachelor","b.sc","b.sc.","b.s","b.s.","btech","bs","bachelor of","mtech","master","ms","msc","mba","phd"])
    edu_score = 1.0 if edu_found else 0.0
    # experience extraction
    years = re.findall(r"(\d+)\s*\+?\s*(?:years|yrs)\b", resume_text.lower())
    years_num = max([int(y) for y in years]) if years else 0
    exp_score = min(years_num / 10.0, 1.0)  # caps at 10+ years
    # aggregate score (weights)
    score = int(100 * (0.6 * match_pct + 0.25 * edu_score + 0.15 * exp_score))
    details = {
        "match_pct": round(match_pct,3),
        "edu_found": edu_found,
        "years_experience": years_num
    }
    return {"score": score, "match_pct": match_pct, "matched": matched, "jd_keywords": jd_keywords, "details": details}

###########################
# LLM-powered scoring
###########################
def llm_score_openai(resume_text, jd_text, model="gpt-3.5-turbo"):
    if openai is None:
        return {"error":"openai package not installed."}
    # get API key from environment or streamlit secrets
    api_key = os.getenv("OPENAI_API_KEY") or (st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else None)
    if not api_key:
        return {"error":"OpenAI API key not found. Set OPENAI_API_KEY as env var or Streamlit secret."}
    openai.api_key = api_key

    system = (
        "You are a helpful assistant that extracts structured information from resumes "
        "and scores how relevant the resume is to a job description. Output VALID JSON only."
    )
    user_prompt = (
        "Resume:\n```\n" + resume_text[:4000] + "\n```\n\n"
        "Job Description:\n```\n" + jd_text[:4000] + "\n```\n\n"
        "Return a JSON object with these fields:\n"
        " - skills: [list of skills found in resume]\n"
        " - education: summarized string\n"
        " - experience_years: int\n"
        " - score: integer 0-100 (higher is better)\n"
        " - matched_skills: list of skills that match job description\n"
        " - reasoning: short text explaining why the score\n        "
        "Make sure output is valid JSON and nothing else."
    )

    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role":"system", "content": system},
                {"role":"user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=800
        )
        text = resp["choices"][0]["message"]["content"].strip()
        parsed = json.loads(text)  # Try parse JSON
        return {"llm_response": parsed}
    except Exception as e:
        return {"error": str(e)}

###########################
# Streamlit UI
###########################
st.title("Automated Resume Relevance Checker")
st.sidebar.header("Options")

mode = st.sidebar.radio("Scoring mode", ("Simple Keyword Match (No API)", "LLM-powered (needs API)"))
model_choice = st.sidebar.selectbox("LLM model (if using LLM)", ("gpt-3.5-turbo", "gpt-4o-mini"), index=0)
st.sidebar.markdown("---")
st.sidebar.write("Hackathon submission checklist:")
st.sidebar.write("- GitHub repo with README")
st.sidebar.write("- Deployed web-app URL")
st.sidebar.write("- 15-30 minute video walkthrough")

col1, col2 = st.columns([1.2, 1])
with col1:
    st.subheader("Upload Candidate Resume")
    uploaded_resume = st.file_uploader("Resume (PDF / DOCX / TXT)", type=["pdf","docx","txt"], key="resume")
    resume_text = extract_text_from_file(uploaded_resume) if uploaded_resume else ""
    if uploaded_resume and not resume_text:
        st.warning("Could not extract text from this resume file.")
    if st.checkbox("Show extracted resume text (for debugging)", value=False):
        st.text_area("Resume text", resume_text, height=300)

with col2:
    st.subheader("Job Description (paste or upload)")
    jd_text_input = st.text_area("Paste the Job Description here", height=250)
    jd_file = st.file_uploader("Or upload JD (optional TXT / PDF)", type=["pdf","txt","docx"], key="jd")
    if jd_file and not jd_text_input:
        jd_text = extract_text_from_file(jd_file)
    else:
        jd_text = jd_text_input

st.markdown("---")
run_btn = st.button("Evaluate Resume ✅")

if run_btn:
    if not resume_text:
        st.error("Please upload a resume file first.")
    elif not jd_text or jd_text.strip()=="":
        st.error("Please paste or upload a Job Description.")
    else:
        with st.spinner("Scoring..."):
            if mode.startswith("Simple"):
                result = simple_score(resume_text, jd_text)
                st.metric("Relevance score (0-100)", result["score"])
                st.subheader("Matched keywords")
                st.write(result["matched"][:100])
                st.subheader("Details")
                st.json(result["details"])
                st.subheader("Job description keywords (extracted)")
                st.write(result["jd_keywords"][:80])
                st.subheader("Suggestions to improve relevance")
                suggestions = []
                if len(result["matched"]) / max(1,len(result["jd_keywords"])) < 0.5:
                    suggestions.append("Add or highlight the skills listed in the job description (skills section or summary).")
                if not result["details"]["edu_found"]:
                    suggestions.append("Make education level explicit (degree, college, graduation year).")
                if result["details"]["years_experience"] < 2:
                    suggestions.append("If applicable, highlight relevant internships or projects to show experience.")
                st.write(suggestions if suggestions else "Looks good!")
            else:
                out = llm_score_openai(resume_text, jd_text, model=model_choice)
                if "error" in out:
                    st.error(f"LLM Error: {out['error']}")
                else:
                    parsed = out["llm_response"]
                    st.metric("LLM Relevance score (0-100)", parsed.get("score","N/A"))
                    st.subheader("Structured output from LLM")
                    st.json(parsed)
                    st.subheader("Suggestions (from LLM)")
                    st.write(parsed.get("reasoning") or "No suggestions returned.")

st.markdown("---")
st.caption("Built for Hackathon: Automated Resume Relevance Check System. Adapt prompts and scoring weights as needed.")
