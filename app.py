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

# Gemini import for optional LLM mode. Simple mode does not need this package.
try:
    from google import genai
except Exception:
    genai = None

st.set_page_config(page_title="Resume Relevance Checker", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #101217;
        --paper: #f3f1eb;
        --muted: #a9adb8;
        --line: rgba(255, 255, 255, 0.12);
        --coral: #ff725e;
    }

    .stApp {
        background:
            radial-gradient(circle at 86% 8%, rgba(255, 114, 94, 0.12), transparent 24rem),
            linear-gradient(145deg, #0d0f14 0%, #151820 55%, #111318 100%);
        color: var(--paper);
    }

    .block-container {
        max-width: 1180px;
        padding: 4.5rem 3rem 4rem;
    }

    [data-testid="stSidebar"] {
        background: rgba(10, 12, 17, 0.92);
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p { font-family: 'Manrope', sans-serif; }

    h1, h2, h3, p, label, button, [data-testid="stMetricValue"] {
        font-family: 'Manrope', sans-serif;
    }

    h1 {
        max-width: 760px;
        margin: 0.4rem 0 0.65rem;
        color: var(--paper);
        font-size: clamp(2.6rem, 6vw, 5.4rem) !important;
        font-weight: 800 !important;
        letter-spacing: -0.06em !important;
        line-height: 0.98 !important;
    }

    h2, h3 { letter-spacing: -0.03em; }

    .eyebrow {
        color: var(--coral);
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .hero-copy {
        max-width: 610px;
        margin-bottom: 3rem;
        color: var(--muted);
        font-size: 1.03rem;
        line-height: 1.65;
    }

    [data-testid="stFileUploader"] section,
    [data-testid="stTextArea"] textarea {
        border: 1px solid var(--line);
        border-radius: 6px;
        background: rgba(255, 255, 255, 0.055);
    }

    [data-testid="stFileUploader"] section:hover { border-color: var(--coral); }

    [data-testid="stButton"] button {
        border: 0;
        border-radius: 5px;
        background: var(--coral);
        color: var(--ink);
        font-weight: 800;
        min-height: 3rem;
        transition: transform 160ms ease, filter 160ms ease;
    }

    [data-testid="stButton"] button:hover {
        border: 0;
        color: var(--ink);
        filter: brightness(1.08);
        transform: translateY(-2px);
    }

    [data-testid="stMetric"] {
        border-top: 2px solid var(--coral);
        padding-top: 0.75rem;
    }

    code, .stCaption { font-family: 'DM Mono', monospace; }
    </style>
    """,
    unsafe_allow_html=True,
)

###########################
# Utility: text extraction
###########################
@st.cache_data(show_spinner=False)
def extract_text_from_file(data, filename):
    """Extract uploaded file bytes; bytes and filename are cacheable inputs."""
    if not data:
        return ""
    fname = filename.lower()
    content = ""
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

def get_uploaded_text(uploaded_file):
    """Convert Streamlit's unhashable UploadedFile to cached byte inputs."""
    if uploaded_file is None:
        return ""
    return extract_text_from_file(uploaded_file.getvalue(), uploaded_file.name)

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
# Authenticity / distinctiveness analysis (no API required)
###########################
# Keep this list explicit so it is easy to maintain for the portfolio project.
CLICHE_PHRASES = [
    "spearheaded", "leveraged", "results-driven", "dynamic", "synergy",
    "proven track record", "self-starter", "team player", "detail-oriented",
    "go-getter", "strategic thinker", "passionate about", "utilized",
    "responsible for", "hardworking", "excellent communication skills",
    "fast-paced environment", "wear many hats", "think outside the box",
    "hit the ground running", "value-add", "circle back", "bandwidth"
]
TOOL_KEYWORDS = [
    "python", "java", "javascript", "typescript", "sql", "excel", "power bi",
    "tableau", "salesforce", "hubspot", "aws", "azure", "gcp", "docker",
    "kubernetes", "git", "github", "react", "node.js", "django", "flask",
    "streamlit", "figma", "jira", "sap", "oracle", "snowflake", "databricks",
    "tensorflow", "pytorch", "linux", "google analytics", "quickbooks",
    "machine learning", "feature engineering", "data cleaning", "exploratory data analysis",
    "data analytics", "data structures and algorithms", "scikit-learn"
]
REQUIREMENT_ALIASES = {
    "data visualization": ["data visualization", "dashboard", "dashboards", "power bi", "tableau", "matplotlib", "seaborn"],
    "machine learning": ["machine learning", "predictive model", "classification", "regression", "scikit-learn"],
    "data analysis": ["data analytics", "exploratory data analysis", "eda", "analyzed data", "data analysis"],
    "api development": ["api", "rest api", "fastapi", "flask", "backend service"],
    "database": ["sql", "mysql", "postgresql", "mongodb", "database"],
    "cloud deployment": ["aws", "azure", "gcp", "cloud", "deployed"],
}
VAGUE_STARTS = ("responsible for", "worked on", "helped with", "assisted with", "participated in")
OUTCOME_VERBS = {
    "achieved", "automated", "built", "created", "decreased", "delivered", "developed",
    "drove", "improved", "increased", "launched", "led", "optimized", "reduced",
    "saved", "streamlined", "transformed", "grew", "implemented", "designed", "analyzed", "trained"
}
SECTION_HEADERS = {
    "technical skills", "skills", "education", "experience", "projects",
    "certifications", "achievements", "summary", "objective", "profile",
    "contact", "interests", "languages", "work experience", "professional experience",
    "projects & experience", "course institution board year cgpa/%", "education details",
    "personal details", "career objective", "about me", "resume summary"
}


def looks_like_bullet(raw_text):
    """Return True only for resume-like accomplishment bullets, not names/headers/skill lists."""
    if raw_text is None:
        return False
    text = re.sub(r"^[\s•\-–—*]+", "", str(raw_text)).strip()
    if not text or len(text) < 12:
        return False

    lower = text.lower()
    compact = re.sub(r"\s+", " ", lower).strip()
    if compact in SECTION_HEADERS:
        return False
    if compact.startswith("technical skills") or compact.startswith("skills"):
        return False
    if any(compact.startswith(prefix) for prefix in ["education", "experience", "projects", "certifications", "achievements", "summary", "objective", "contact", "profile", "languages", "interests"]):
        return False

    words = re.findall(r"[A-Za-z][A-Za-z'./#&+-]*", text)
    if not words:
        return False

    # Exclude names and short title-case labels (e.g., "Devireddy Rohith Reddy").
    if len(words) <= 5 and sum(1 for w in words if w[0].isupper()) >= max(2, len(words) - 1):
        if not re.search(r"\b[a-z]+(?:\s+[a-z]+){1,}\b", text):
            return False

    # Exclude table header rows such as "Course Institution Board Year CGPA/%".
    if len(words) >= 4 and sum(1 for w in words if w[0].isupper()) >= len(words) - 1:
        if not re.search(r"\b[a-z]+(?:\s+[a-z]+){1,}\b", text):
            return False

    # Reject obvious fragments like "neering and Science Present" created by splitting on a hyphen mid-line.
    has_verb = bool(re.search(r"\b(?:built|developed|created|automated|improved|reduced|increased|optimized|launched|delivered|designed|implemented|managed|led|trained|integrated|achieved|analyzed|streamlined|saved|grew|worked|worked on|used|using|enabled|built|created|handled|supported|migrated|optimized|boosted|cut|improved|reduced)\b", lower))
    has_metric = bool(re.search(r"(?:[$€£₹]\s*)?\d+(?:[,.]\d+)?\s*(?:%|\+|x\b|k\b|m\b)?", text, re.I))
    has_tool = any(tool in lower for tool in [t for t in TOOL_KEYWORDS if len(t) > 2])
    has_context = bool(re.search(r"\b(?:using|with|via|through|by|in)\b", lower))
    if not (has_verb or has_metric or has_tool or has_context):
        return False

    # Ensure there's at least some sentence-like lower-case text, not just a label with no verb.
    if not re.search(r"\b[a-z]+(?:\s+[a-z]+){1,}\b", text):
        return False

    return True


def split_resume_lines(resume_text):
    """Return resume bullet-like lines after filtering out names, headers, and skill lists."""
    lines = []
    for line_number, raw_line in enumerate(resume_text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue

        # Only split on true bullet markers at the start of a line or section, not on hyphenated words.
        segments = [stripped]
        if re.search(r"^(?:[•▪‣◦*]|\d+\.|\d+\)|[-–—]\s+)", stripped):
            segments = re.split(r"(?:^|\n)\s*(?:[•▪‣◦*]|\d+\.|\d+\)|[-–—]\s+)", stripped)

        for segment in segments:
            text = re.sub(r"^[\s•\-–—*]+", "", segment).strip()
            if text and looks_like_bullet(text):
                lines.append({"line_number": line_number, "text": text})
    return lines


def analyse_authenticity(resume_text):
    """Calculate cliche density and a 0-3 evidence/specificity score per line."""
    lines = split_resume_lines(resume_text)
    cliche_hits, bullet_scores = [], []
    for item in lines:
        text, lower = item["text"], item["text"].lower()
        cliches = [phrase for phrase in CLICHE_PHRASES if phrase in lower]
        cliche_hits.extend({"phrase": phrase, **item} for phrase in cliches)
        has_metric = bool(re.search(r"(?:[$€£₹]\s*)?\d+(?:[,.]\d+)?\s*(?:%|\+|x\b|k\b|m\b)?", text, re.I))
        has_tool = any(re.search(r"(?<!\w)" + re.escape(tool) + r"(?!\w)", lower) for tool in TOOL_KEYWORDS)
        starts_vague = lower.startswith(VAGUE_STARTS)
        # A lightweight heuristic: allow a short lead-in such as "In 2023, led...",
        # but do not treat vague responsibility starters as outcome statements.
        opening_words = re.findall(r"[a-z]+", lower)[:4]
        has_outcome = any(word in OUTCOME_VERBS for word in opening_words) and not starts_vague
        bullet_scores.append({**item, "score": int(has_metric) + int(has_tool) + int(has_outcome),
                              "cliches": cliches, "has_metric": has_metric,
                              "has_tool": has_tool, "has_outcome": has_outcome})
    average = sum(item["score"] for item in bullet_scores) / len(bullet_scores) if bullet_scores else 0
    weakest = sorted(bullet_scores, key=lambda item: (item["score"], -len(item["cliches"])))[:5]
    return {"lines": lines, "cliche_hits": cliche_hits, "bullet_scores": bullet_scores,
            "cliche_density": len(cliche_hits) / len(lines) if lines else 0,
            "average_specificity": average, "weakest": weakest}

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
    evidence_rows, evidence_coverage = build_requirement_evidence(resume_text, jd_text)
    # Evidence coverage is weighted above raw token overlap so repeated keywords
    # cannot substitute for a supporting resume line.
    score = int(100 * (0.55 * evidence_coverage + 0.15 * match_pct + 0.2 * edu_score + 0.1 * exp_score))
    details = {
        "match_pct": round(match_pct,3),
        "evidence_coverage": round(evidence_coverage, 3),
        "edu_found": edu_found,
        "years_experience": years_num
    }
    return {"score": score, "match_pct": match_pct, "matched": matched, "jd_keywords": jd_keywords, "details": details}


REQUIREMENT_STOPWORDS = {
    "job", "role", "work", "team", "candidate", "candidates", "skills", "skill",
    "experience", "years", "year", "required", "preferred", "ability", "strong",
    "knowledge", "responsibilities", "responsibility", "including", "working", "using",
    "applications", "application", "apis", "api", "fresher", "develop", "developed",
    "clean", "data", "feature", "engineering", "power"
}


def build_requirement_evidence(resume_text, jd_text, limit=15):
    """Match requirements to exact or concept-level resume evidence."""
    resume_lines = [line.strip() for line in resume_text.splitlines() if line.strip()]
    jd_lower = jd_text.lower()
    phrase_requirements = [tool for tool in TOOL_KEYWORDS
                           if re.search(r"(?<![A-Za-z0-9])" + re.escape(tool) + r"(?![A-Za-z0-9])", jd_lower)]
    concept_requirements = [concept for concept, aliases in REQUIREMENT_ALIASES.items()
                            if any(re.search(r"(?<![A-Za-z0-9])" + re.escape(alias) + r"(?![A-Za-z0-9])", jd_lower)
                                   for alias in aliases)]
    phrase_words = {word for phrase in phrase_requirements if " " in phrase for word in phrase.split()}
    phrase_words.update(word for concept in concept_requirements for word in concept.split())
    keyword_requirements = [keyword.replace("-", " ") for keyword in extract_keywords_basic(jd_text, top_k=40)
                            if len(keyword) > 2 and keyword not in REQUIREMENT_STOPWORDS and keyword not in phrase_words]
    requirements = list(dict.fromkeys(concept_requirements + phrase_requirements + keyword_requirements))
    rows = []
    for requirement in requirements[:limit]:
        aliases = REQUIREMENT_ALIASES.get(requirement, [requirement])
        evidence = None
        matched_alias = None
        for alias in aliases:
            pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(alias).replace(r"\ ", r"[ -]+") + r"(?![A-Za-z0-9])", re.I)
            evidence = next((line for line in resume_lines if pattern.search(line)), None)
            if evidence:
                matched_alias = alias
                break
        if evidence and matched_alias == requirement:
            match_type, evidence_score = "Exact evidence", 1.0
        elif evidence:
            match_type, evidence_score = "Concept match", 0.8
        else:
            match_type, evidence_score = "No evidence", 0.0
        rows.append({
            "Requirement": requirement,
            "Status": "Supported" if evidence else "Not found",
            "Resume evidence": evidence or "No matching resume line found",
            "Match type": match_type,
            "Evidence score": evidence_score
        })
    coverage = sum(row["Evidence score"] for row in rows) / len(rows) if rows else 0
    return rows, coverage


def render_requirement_evidence(resume_text, jd_text):
    """Render transparent requirement coverage linked to resume text."""
    rows, coverage = build_requirement_evidence(resume_text, jd_text)
    st.subheader("Proof of fit")
    st.caption("A requirement counts as supported only when the resume contains a matching line. No claims are generated here.")
    if not rows:
        st.info("No meaningful requirements could be extracted from the job description.")
        return

    supported = [row for row in rows if row["Status"] == "Supported"]
    missing = [row for row in rows if row["Status"] == "Not found"]
    metric_one, metric_two, metric_three = st.columns(3)
    metric_one.metric("Evidence coverage", f"{coverage:.0%}")
    metric_two.metric("Requirements supported", len(supported))
    metric_three.metric("Evidence gaps", len(missing))
    st.progress(coverage)

    if supported:
        st.markdown("**Evidence found**")
        for row in supported:
            with st.expander(f"✓ {row['Requirement'].title()}  ·  {row['Match type']}"):
                st.write(f"“{row['Resume evidence']}”")

    if missing:
        st.markdown("**Evidence gaps to fix**")
        for row in missing:
            with st.expander(f"! {row['Requirement'].title()}  ·  Needs verification"):
                st.write("This requirement was not found in the resume.")
                st.caption("Add a truthful project, responsibility, tool, or result that demonstrates it, if applicable.")


def looks_like_error_log(text):
    """Detect pasted runtime errors that are not job descriptions."""
    lower = text.lower()
    markers = [
        "traceback (most recent call last)", "systemerror:", "attributeerror:",
        "filenotfounderror:", "modulenotfounderror:", "typeerror:",
        "exception:", "streamlit.errors", "site-packages\\", "site-packages/"
    ]
    return sum(marker in lower for marker in markers) >= 2

###########################
# Gemini-powered scoring
###########################
# The app intentionally uses one model so scoring is predictable across runs.
GEMINI_MODEL = "gemini-3.8-flash"

# Gemini structured outputs prevent missing
# fields or prose around the JSON, making the UI output dependable.
RELEVANCE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "resume_relevance_assessment",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "skills": {"type": "array", "items": {"type": "string"}},
                "education": {"type": "string"},
                "experience_years": {"type": "integer"},
                "score": {"type": "integer", "minimum": 0, "maximum": 100},
                "matched_skills": {"type": "array", "items": {"type": "string"}},
                "reasoning": {"type": "string"},
            },
            "required": ["skills", "education", "experience_years", "score", "matched_skills", "reasoning"],
        },
    },
}
REWRITE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "resume_bullet_rewrites",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "rewrites": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"id": {"type": "integer"}, "rewrite": {"type": "string"}},
                        "required": ["id", "rewrite"],
                    },
                },
            },
            "required": ["rewrites"],
        },
    },
}

def get_gemini_client():
    """Safely get a Gemini key from env vars or Streamlit secrets."""
    if genai is None:
        return None, "google-genai package not installed. Run: pip install google-genai"
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        try:
            api_key = str(st.secrets.get("GEMINI_API_KEY", "")).strip()
        except Exception:
            api_key = None
    if not api_key:
        return None, "Gemini API key not found. Set GEMINI_API_KEY as an environment variable or Streamlit secret."
    try:
        return genai.Client(api_key=api_key), None
    except Exception as error:
        return None, f"Could not initialize the Gemini client: {error}"

def parse_json_response(text):
    """Parse JSON returned as plain text or inside a Markdown code fence."""
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Gemini returned no JSON object")
        parsed = json.loads(raw[start:end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Gemini returned an invalid JSON object")
    return parsed

def llm_score_gemini(resume_text, jd_text):
    client, error = get_gemini_client()
    if error:
        return {"error": error}

    user_prompt = (
        "Assess how relevant this resume is to the job description. Use only evidence in the resume; "
        "do not invent skills, education, or experience.\n\n"
        "Resume:\n```\n" + resume_text[:4000] + "\n```\n\n"
        "Job Description:\n```\n" + jd_text[:4000] + "\n```\n\n"
        "Return JSON only. Keep reasoning under 80 words and lists under 12 items."
    )

    try:
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_prompt,
            config={
                "temperature": 0.2,
                "max_output_tokens": 1400,
                "response_mime_type": "application/json",
                "response_schema": RELEVANCE_RESPONSE_FORMAT["json_schema"]["schema"],
            },
        )
        parsed = parse_json_response(resp.text)
        return {"llm_response": parsed}
    except Exception as first_error:
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=user_prompt + (
                    '\nUse exactly this compact JSON shape: '
                    '{"skills":[],"education":"","experience_years":0,"score":0,'
                    '"matched_skills":[],"reasoning":""}. JSON only.'
                ),
                config={"temperature": 0.1, "max_output_tokens": 1200},
            )
            return {"llm_response": parse_json_response(response.text)}
        except Exception as retry_error:
            return {"error": f"Gemini structured output failed: {retry_error} (initial error: {first_error})"}

def llm_rewrite_bullets(bullets):
    """Get fact-preserving rewrites for only the weak/cliche-heavy bullets."""
    if not bullets:
        return {}, None
    client, error = get_gemini_client()
    if error:
        return {}, error
    numbered = "\n".join(f"{bullet['line_number']}. {bullet['text']}" for bullet in bullets)
    prompt = (
        "Rewrite each resume bullet below to sound more specific and credible. Do NOT invent facts, "
        "numbers, tools, or achievements not implied in the original. Only make existing content more "
        "concrete and remove generic AI-sounding phrasing. Keep each rewrite to one line. "
        "Return one rewrite for every input line.\n\n" + numbered
    )
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "temperature": 0.3,
                "max_output_tokens": 700,
                "response_mime_type": "application/json",
                "response_schema": REWRITE_RESPONSE_FORMAT["json_schema"]["schema"],
            },
        )
        data = parse_json_response(response.text)
        return {item.get("id"): item.get("rewrite", "") for item in data.get("rewrites", [])}, None
    except Exception as first_error:
        # Retry without schema enforcement if structured output is rejected.
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt + (
                    "\nReturn JSON only, with exactly this shape: "
                    '{"rewrites":[{"id":1,"rewrite":"text"}]}.'
                ),
                config={"temperature": 0.2, "max_output_tokens": 700},
            )
            data = parse_json_response(response.text)
            return {item.get("id"): item.get("rewrite", "") for item in data.get("rewrites", [])}, None
        except Exception as retry_error:
            return {}, f"Gemini rewrite failed after retry: {retry_error} (initial error: {first_error})"

def render_authenticity_check(analysis, rewrites=None):
    """Show the shared local analysis beneath either relevance score."""
    st.subheader("🔍 Authenticity & Distinctiveness Check")
    density = analysis["cliche_density"]
    average = analysis["average_specificity"]
    # Demo-only heuristic baseline: 0.15 cliches/bullet and 1.5/3 specificity.
    # This is not a statistically calibrated percentile or a claim about a real dataset.
    heuristic_percentile = round(min(99, max(1, 50 + (0.15 - density) * 150 + (average - 1.5) * 20)))
    c1, c2 = st.columns(2)
    c1.metric("Cliché Density", f"{density:.0%}", help="Cliché phrase occurrences divided by resume bullets/lines.")
    c2.metric("Average Specificity Score", f"{average:.1f}/3")
    c2.progress(int(round(average / 3 * 100)))
    if average < 1.5:
        st.warning("Most accomplishment bullets need stronger evidence. Add a measurable result, name the tool or method used, and begin with a clear action or outcome verb.")
    elif average < 2.5:
        st.info("Your bullets have some evidence. Strengthen the weaker ones with a measurable result or a clearer outcome.")
    else:
        st.success("Most bullets include concrete evidence, tools, and outcomes.")
    st.caption(
        f"Demo heuristic (not a validated percentile): this resume scores more specific/less generic than about "
        f"{heuristic_percentile}% using the app's fixed baseline."
    )

    if len(analysis["bullet_scores"]) < 3:
        st.info("No accomplishment-style bullets were detected — this resume may rely mostly on skills lists rather than project/experience descriptions. Consider adding specific project or experience bullets with measurable outcomes.")
        return

    if analysis["cliche_hits"]:
        st.markdown("**Flagged cliché phrases**")
        for hit in analysis["cliche_hits"]:
            st.write(f"• `{hit['phrase']}` — line {hit['line_number']}: {hit['text']}")
    else:
        st.success("No phrases from the maintained cliché list were found.")

    st.markdown("**Weakest bullets**")
    if not analysis["weakest"]:
        st.info("No resume bullets/lines were available to score.")
        return
    rows = []
    for index, bullet in enumerate(analysis["weakest"], start=1):
        evidence = ", ".join(label for label, found in [
            ("metric", bullet["has_metric"]), ("tool", bullet["has_tool"]), ("outcome", bullet["has_outcome"])
        ] if found) or "no metric, tool, or concrete outcome detected"
        missing = ", ".join(label for label, found in [
            ("measurable result", not bullet["has_metric"]),
            ("named tool or method", not bullet["has_tool"]),
            ("clear outcome verb", not bullet["has_outcome"])
        ] if found) or "none"
        rows.append({"Line": bullet["line_number"], "Specificity /3": bullet["score"],
                     "Original bullet": bullet["text"], "Evidence found": evidence,
                     "What to add": missing,
                     "Suggested rewrite": (rewrites or {}).get(bullet["line_number"], "Gemini mode required for a rewrite")})
    for row in rows:
        title = f"Line {row['Line']} · Specificity {row['Specificity /3']}/3"
        with st.expander(title):
            st.markdown(f"**Original bullet**  \n{row['Original bullet']}")
            st.caption(f"Evidence found: {row['Evidence found']}")
            st.caption(f"What to add: {row['What to add']}")
            st.markdown(f"**Suggested rewrite**  \n{row['Suggested rewrite']}")


def render_resume_result(resume_name, resume_text, jd_text, mode):
    """Score one resume and render its evidence-focused review."""
    authenticity = analyse_authenticity(resume_text)
    with st.spinner(f"Scoring {resume_name}..."):
        if mode.startswith("Simple"):
            result = simple_score(resume_text, jd_text)
            st.metric("Relevance score (0-100)", result["score"])
            st.subheader("Matched keywords")
            st.write(result["matched"][:100])
            st.subheader("Details")
            st.json(result["details"])
            st.subheader("Job description keywords (extracted)")
            st.write(result["jd_keywords"][:80])
            suggestions = []
            if len(result["matched"]) / max(1, len(result["jd_keywords"])) < 0.5:
                suggestions.append("Add or highlight the skills listed in the job description.")
            if not result["details"]["edu_found"]:
                suggestions.append("Make education level explicit.")
            if result["details"]["years_experience"] < 2:
                suggestions.append("If applicable, highlight relevant internships or projects.")
            st.subheader("Suggestions to improve relevance")
            st.write(suggestions if suggestions else "Looks good!")
            render_requirement_evidence(resume_text, jd_text)
            render_authenticity_check(authenticity)
            return result["score"], authenticity

        out = llm_score_gemini(resume_text, jd_text)
        if "error" in out:
            fallback = simple_score(resume_text, jd_text)
            st.warning(f"Gemini was unavailable, so the transparent local evidence score was used. Details: {out['error']}")
            st.metric("Local evidence score (0-100)", fallback["score"])
            st.json(fallback["details"])
            render_requirement_evidence(resume_text, jd_text)
            render_authenticity_check(authenticity)
            return fallback["score"], authenticity
        parsed = out["llm_response"]
        st.metric("Gemini Relevance score (0-100)", parsed.get("score", "N/A"))
        st.subheader("Structured output from Gemini")
        st.json(parsed)
        st.subheader("Suggestions (from Gemini)")
        st.write(parsed.get("reasoning") or "No suggestions returned.")
        render_requirement_evidence(resume_text, jd_text)
        flagged = [bullet for bullet in authenticity["bullet_scores"]
                   if bullet["score"] < 3 or bullet["cliches"]]
        rewrites, rewrite_error = llm_rewrite_bullets(flagged)
        if rewrite_error:
            st.info(f"Authenticity rewrites unavailable: {rewrite_error}")
        render_authenticity_check(authenticity, rewrites)
        return parsed.get("score", 0), authenticity

###########################
# Streamlit UI
###########################
st.markdown('<div class="eyebrow">SIGNAL / 01 &nbsp;·&nbsp; EVIDENCE-LED REVIEW</div>', unsafe_allow_html=True)
st.title("Resume relevance, with receipts.")
st.markdown(
    '<p class="hero-copy">Compare a candidate to the role, see which requirements are actually supported, and find the vague bullets that need stronger evidence.</p>',
    unsafe_allow_html=True,
)
st.sidebar.header("Options")

mode = st.sidebar.radio("Scoring mode", ("Simple Keyword Match (No API)", "Gemini-powered (needs API)"))
st.sidebar.caption("AI model: Gemini 3.8 Flash")

col1, col2 = st.columns([1.2, 1])
with col1:
    st.subheader("Upload Candidate Resumes")
    uploaded_resumes = st.file_uploader(
        "Select one or many resumes (PDF / DOCX / TXT)",
        type=["pdf", "docx", "txt"], accept_multiple_files=True, key="resumes"
    )
    resume_text = get_uploaded_text(uploaded_resumes[0]) if uploaded_resumes else ""
    if uploaded_resumes:
        st.caption(f"{len(uploaded_resumes)} resume(s) ready for comparison")
    if uploaded_resumes and not resume_text:
        st.warning("Could not extract text from the first resume file.")
    if st.checkbox("Show extracted resume text (for debugging)", value=False):
        st.text_area("Resume text", resume_text, height=300)

with col2:
    st.subheader("Job Description (paste or upload)")
    jd_text_input = st.text_area("Paste the Job Description here", height=250)
    jd_file = st.file_uploader("Or upload JD (optional TXT / PDF)", type=["pdf","txt","docx"], key="jd")
    if jd_file and not jd_text_input:
        jd_text = get_uploaded_text(jd_file)
    else:
        jd_text = jd_text_input

st.markdown("---")
run_btn = st.button("Evaluate Resume ✅")

if run_btn:
    if not uploaded_resumes:
        st.error("Please upload at least one resume file.")
    elif not jd_text or jd_text.strip()=="":
        st.error("Please paste or upload a Job Description.")
    elif looks_like_error_log(jd_text):
        st.error("The Job Description looks like a Python error log, not a job posting. Paste the role's responsibilities and requirements, then evaluate again.")
    else:
        scored = []
        for uploaded_resume in uploaded_resumes:
            candidate_text = get_uploaded_text(uploaded_resume)
            if not candidate_text:
                st.warning(f"Could not extract text from {uploaded_resume.name}; skipped.")
                continue
            score, authenticity = render_resume_result(uploaded_resume.name, candidate_text, jd_text, mode)
            scored.append({"Resume": uploaded_resume.name, "Score": score,
                           "Average specificity /3": round(authenticity["average_specificity"], 2),
                           "Cliché density": f"{authenticity['cliche_density']:.0%}"})

        if len(scored) > 1:
            st.markdown("## Candidate ranking")
            st.caption("Ranked by job relevance, with authenticity signals shown beside the score.")
            for rank, candidate in enumerate(sorted(scored, key=lambda item: item["Score"], reverse=True), start=1):
                st.markdown(
                    f"**{rank}. {candidate['Resume']}**  ·  Score: `{candidate['Score']}`  ·  "
                    f"Specificity: `{candidate['Average specificity /3']}/3`  ·  "
                    f"Cliché density: `{candidate['Cliché density']}`"
                )

###########################
# Quick bullet filter tests
###########################
def test_bullet_filter():
    samples = [
        "Built a churn prediction model using scikit-learn, achieving 89% accuracy",
        "Automated the QA handoff process in Python, reducing cycle time by 40%",
        "Devireddy Rohith Reddy",
        "Course Institution Board Year CGPA/%",
        "Technical Skills",
        "AI & Machine Learning Scikit-learn, Machine Learning, Feature Engineering",
        "B.Tech - Engineering and Science Present",
    ]
    for sample in samples:
        print(f"{sample!r} -> {looks_like_bullet(sample)}")


# Optional quick verification from a terminal:
# test_bullet_filter()

st.markdown("---")
