# Fair Copy - Local

Requires internet access (to download the model on first run, and to reach the article-parsing API for URL input)

```bash
cd backend
pip install -r requirements.txt
python -m spacy download en_core_web_sm
uvicorn main:app --reload
```

Then open `frontend/index.html` directly in a browser. It talks to the backend at `http://127.0.0.1:8000`.
