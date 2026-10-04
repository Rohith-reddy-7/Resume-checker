# Resume Relevance Checker

A Streamlit app that compares resumes with job descriptions using transparent local evidence scoring or Gemini 3.8 Flash.

## Run locally

```powershell
$env:GEMINI_API_KEY = "your-gemini-api-key"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The simple keyword mode works without an API key.

## Deploy with Streamlit Community Cloud

1. Create a GitHub repository and upload `app.py`, `requirements.txt`, `.gitignore`, and this `README.md`.
2. Open [share.streamlit.io](https://share.streamlit.io) and choose **Create app**.
3. Select the repository, branch, and set the main file to `app.py`.
4. In **Advanced settings**, add this secret:

```toml
GEMINI_API_KEY = "your-gemini-api-key"
```

5. Deploy the app.

Never commit an API key to GitHub. Use Streamlit Cloud secrets or an environment variable.
