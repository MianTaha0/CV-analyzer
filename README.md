# CV Uploader & Intelligent Search (Flask + MongoDB)

Features:
- Upload CVs (PDF/DOCX/TXT), auto-parse name/email/skills/experience
- Store in MongoDB, with regex and text search
- Vector-based ranking (lightweight local embedding) + optional OpenAI embeddings
- AI search endpoint and chat-style query
- Voice search in browser using Web Speech API

## Quick Start

1) Configure environment

- Copy `.env.example` to `.env` and adjust values as needed.

2) Start MongoDB

- Preferred: install Docker and run `docker compose up -d`
- Or install MongoDB locally and ensure it runs on `mongodb://localhost:27017`

3) Install Python deps

This environment is PEP 668 managed. Use one of:
- System override (quick):
  `pip3 install --break-system-packages -r requirements.txt`
- Or create venv (requires venv package):
  `apt install -y python3.13-venv && python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt`

4) Run the app

`python3 run.py`

Open `http://localhost:5000`.

## API

- POST `/api/upload` (multipart form) => { id, name, email, phone, skills, total_experience_years, file_name, file_type }
- GET `/api/search?name=&email=&min_experience=&text=` => { results: [...] }
- POST `/api/ai/search` JSON { query, top_k? } => ranked results + reasons
- POST `/api/chat` JSON { message } => reply + results

## Notes
- Set `EMBEDDING_PROVIDER=openai` and `OPENAI_API_KEY` to use OpenAI embeddings.
- Voice search requires a Chromium-based browser with Web Speech API support.