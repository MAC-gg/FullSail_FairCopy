# Fair Copy - RnD Proof of Concept

Requires internet access (to download the model on first run, and to reach the article-parsing API for URL input)

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

First request will download `facebook/bart-large-mnli` (~1.6GB)

Then open `frontend/index.html` directly in a browser. It talks to the backend at `http://127.0.0.1:8000`.
