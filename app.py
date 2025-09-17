import os
import io
import re
import json
import math
import time
import uuid
import base64
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename

from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING, TEXT
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError
from bson import ObjectId

from concurrent.futures import ThreadPoolExecutor

# External parsing libs
try:
    from pdfminer.high_level import extract_text as pdf_extract_text
except Exception:  # pragma: no cover - optional at runtime
    pdf_extract_text = None  # type: ignore

try:
    from docx import Document  # python-docx
except Exception:  # pragma: no cover - optional at runtime
    Document = None  # type: ignore

try:  # optional legacy .doc support
    import textract  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    textract = None  # type: ignore

try:
    from openai import OpenAI
except Exception as e:  # pragma: no cover
    raise RuntimeError("openai package is required. Install with `pip install openai`.") from e


# ------------------------------------------------------------
# App and Config
# ------------------------------------------------------------

load_dotenv()

app = Flask(__name__)
CORS(app)


# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("cv-analyzer")


# Security and upload constraints
MAX_CONTENT_LENGTH_MB = float(os.getenv("MAX_CONTENT_LENGTH_MB", "25"))
app.config["MAX_CONTENT_LENGTH"] = int(MAX_CONTENT_LENGTH_MB * 1024 * 1024)

ALLOWED_EXTENSIONS = {"pdf", "doc", "docx"}


# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not set. Routes depending on OpenAI will fail.")

OPENAI_MODEL_PARsing = os.getenv("OPENAI_MODEL_PARSING", "gpt-4o-mini")
OPENAI_MODEL_RANKING = os.getenv("OPENAI_MODEL_RANKING", "gpt-4o-mini")
OPENAI_MODEL_TTS = os.getenv("OPENAI_MODEL_TTS", "gpt-4o-mini-tts")
OPENAI_MODEL_EMBEDDING = os.getenv("OPENAI_MODEL_EMBEDDING", "text-embedding-3-small")

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# MongoDB configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "cv_analyzer")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "candidates")

mongo_client = MongoClient(MONGO_URI)
db = mongo_client[MONGO_DB_NAME]
candidates_col: Collection = db[MONGO_COLLECTION]


# Vector Search placeholder toggle
USE_ATLAS_VECTOR_SEARCH = os.getenv("USE_ATLAS_VECTOR_SEARCH", "false").lower() == "true"
VECTOR_FIELD = "embedding"


# Ensure indexes
def _ensure_indexes() -> None:
    try:
        candidates_col.create_index([("email", ASCENDING)], unique=True, sparse=True, name="uniq_email")
    except Exception as e:
        logger.warning("Failed to create unique index on email: %s", e)
    try:
        candidates_col.create_index([("role", ASCENDING)], name="idx_role")
    except Exception as e:
        logger.warning("Failed to create index on role: %s", e)
    try:
        # Text index to support keyword search
        candidates_col.create_index([("raw_text", TEXT), ("skills", TEXT), ("role", TEXT)], name="text_all")
    except Exception as e:
        logger.warning("Failed to create text index: %s", e)


_ensure_indexes()


# Background worker for embeddings
EXECUTOR_MAX_WORKERS = int(os.getenv("EXECUTOR_MAX_WORKERS", "4"))
executor = ThreadPoolExecutor(max_workers=EXECUTOR_MAX_WORKERS)


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------


def json_response(data: Any, status_code: int = 200) -> Response:
    return Response(
        response=json.dumps(data, ensure_ascii=False, separators=(",", ":")),
        status=status_code,
        mimetype="application/json",
    )


def error_response(message: str, status_code: int = 400, details: Optional[Dict[str, Any]] = None) -> Response:
    payload: Dict[str, Any] = {"error": message}
    if details:
        payload["details"] = details
    return json_response(payload, status_code)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def read_file_bytes(file_storage) -> bytes:
    file_stream = file_storage.stream if hasattr(file_storage, "stream") else file_storage
    file_stream.seek(0)
    data = file_stream.read()
    file_stream.seek(0)
    return data


def extract_text_from_pdf(file_bytes: bytes) -> str:
    if not pdf_extract_text:
        raise RuntimeError("pdfminer.six is required for PDF extraction. Install `pdfminer.six`.")
    with io.BytesIO(file_bytes) as bio:
        return pdf_extract_text(bio) or ""


def extract_text_from_docx(file_bytes: bytes) -> str:
    if not Document:
        raise RuntimeError("python-docx is required for DOCX extraction. Install `python-docx`.")
    with io.BytesIO(file_bytes) as bio:
        doc = Document(bio)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(paragraphs)


def extract_text_from_doc(file_bytes: bytes) -> str:
    # Legacy .doc support via textract if available
    if not textract:
        raise RuntimeError(".doc extraction requires `textract`. Install `textract` or convert to DOCX.")
    with io.BytesIO(file_bytes) as bio:
        content = textract.process("input.doc", input_encoding=None, extension="doc", stream=bio)  # type: ignore
        return content.decode("utf-8", errors="replace")


def extract_text_from_file(file_storage) -> str:
    """Extract raw text from an uploaded file supporting PDF, DOC, DOCX.

    Args:
        file_storage: Werkzeug FileStorage-like object

    Returns:
        Extracted plain text string
    """
    filename = secure_filename(file_storage.filename or "")
    if not filename or not allowed_file(filename):
        raise ValueError("Unsupported or missing file. Allowed: pdf, doc, docx")

    ext = filename.rsplit(".", 1)[1].lower()
    data = read_file_bytes(file_storage)

    if ext == "pdf":
        text = extract_text_from_pdf(data)
    elif ext == "docx":
        text = extract_text_from_docx(data)
    elif ext == "doc":
        text = extract_text_from_doc(data)
    else:
        raise ValueError("Unsupported file extension")

    return (text or "").strip()


def _default_profile_schema() -> Dict[str, Any]:
    return {
        "name": "",
        "email": "",
        "phone": "",
        "skills": [],
        "experience": [],
        "education": [],
        "location": "",
        "total_experience": {"years": 0, "formatted": ""},
        "role": "",
    }


def _ensure_list_of_strings(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: List[str] = []
        for v in value:
            if isinstance(v, str):
                s = v.strip()
                if s:
                    out.append(s)
            else:
                out.append(str(v))
        return out
    if isinstance(value, str):
        # Split on commas if it's a string
        parts = [p.strip() for p in re.split(r",|\n|;", value) if p and p.strip()]
        return parts
    return [str(value)]


def validate_and_normalize_profile(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize the parsed profile to match schema."""
    schema = _default_profile_schema()
    result: Dict[str, Any] = {}
    for key in schema.keys():
        result[key] = data.get(key, schema[key])
    # Enforce skills is list of strings
    result["skills"] = _ensure_list_of_strings(result.get("skills"))
    # Coerce total_experience
    te = result.get("total_experience") or {}
    if not isinstance(te, dict):
        te = {"years": 0, "formatted": str(te)}
    years = te.get("years", 0)
    try:
        years_num = float(years)
    except Exception:
        years_num = 0.0
    formatted = te.get("formatted") or f"{years_num:.1f} years"
    result["total_experience"] = {"years": years_num, "formatted": formatted}
    # Simple email normalization
    if isinstance(result.get("email"), str):
        result["email"] = result["email"].strip().lower()
    else:
        result["email"] = ""
    # Trim strings
    for k, v in list(result.items()):
        if isinstance(v, str):
            result[k] = v.strip()
    return result


def parse_resume_with_gpt(raw_text: str, max_retries: int = 2) -> Dict[str, Any]:
    """Use OpenAI to convert raw resume text into structured JSON.

    Retries on invalid JSON and enforces schema. Skills are always list of strings.
    """
    if not openai_client:
        raise RuntimeError("OpenAI client not configured")

    system_prompt = (
        "You are a precise resume parser. Extract fields and return ONLY valid minified JSON. "
        "Ensure `skills` is an array of strings. Use empty defaults when uncertain."
    )
    user_prompt = (
        "Convert the following resume text into JSON with keys: name, email, phone, skills (array of strings), "
        "experience (array), education (array), location, total_experience {years, formatted}, role.\n\n" + raw_text
    )

    last_error: Optional[str] = None
    for attempt in range(max_retries + 1):
        try:
            chat = openai_client.chat.completions.create(
                model=OPENAI_MODEL_PARsing,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            content = chat.choices[0].message.content or "{}"
            data = json.loads(content)
            profile = validate_and_normalize_profile(data)
            return profile
        except Exception as e:
            last_error = str(e)
            logger.warning("Parse attempt %s failed: %s", attempt + 1, last_error)
            # Fallback: try to extract JSON braces from free-form text
            try:
                content_text = locals().get("content", "")
                match = re.search(r"\{[\s\S]*\}", content_text)
                if match:
                    data = json.loads(match.group(0))
                    profile = validate_and_normalize_profile(data)
                    return profile
            except Exception:
                pass
            time.sleep(0.5)

    raise RuntimeError(f"Failed to parse resume JSON: {last_error}")


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def get_text_for_embedding(profile: Dict[str, Any], raw_text: str) -> str:
    # Prefer raw resume text, but enrich with structured summary
    skills = ", ".join(profile.get("skills", [])[:50])
    role = profile.get("role", "")
    name = profile.get("name", "")
    exp_years = profile.get("total_experience", {}).get("years", 0)
    header = f"Name: {name}\nRole: {role}\nSkills: {skills}\nExperienceYears: {exp_years}\n\n"
    return header + (raw_text or "")


def compute_and_store_embedding(candidate_id: ObjectId, text: str) -> None:
    if not openai_client:
        logger.warning("OpenAI client missing. Skipping embedding for %s", candidate_id)
        return
    if not text:
        logger.warning("Empty text for embedding for %s", candidate_id)
        return
    try:
        emb = openai_client.embeddings.create(model=OPENAI_MODEL_EMBEDDING, input=text)
        vector = emb.data[0].embedding  # type: ignore
        candidates_col.update_one({"_id": candidate_id}, {"$set": {VECTOR_FIELD: vector, "updated_at": datetime.utcnow()}})
    except Exception as e:
        logger.error("Failed to compute/store embedding for %s: %s", candidate_id, e)


def schedule_embedding(profile: Dict[str, Any], raw_text: str, candidate_id: ObjectId) -> None:
    text = get_text_for_embedding(profile, raw_text)
    executor.submit(compute_and_store_embedding, candidate_id, text)


def shortlist_candidates_for_job_text(job_text: str, limit: int = 25) -> List[Dict[str, Any]]:
    """Build a lightweight shortlist using text and semantic search when available."""
    query = (job_text or "").strip()
    if not query:
        return []

    # Try text search first
    docs: List[Dict[str, Any]] = []
    try:
        # Use $text when index exists
        results = candidates_col.find(
            {"$text": {"$search": query}},
            {"score": {"$meta": "textScore"}, "name": 1, "role": 1, "skills": 1, "total_experience": 1},
        ).sort([("score", {"$meta": "textScore"})]).limit(limit)
        docs = list(results)
    except Exception:
        # Fallback: regex OR across tokens
        tokens = [re.escape(t) for t in re.findall(r"\w+", query) if t]
        regex = re.compile("|".join(tokens), re.IGNORECASE) if tokens else None
        if regex:
            docs = list(
                candidates_col.find(
                    {"$or": [
                        {"raw_text": {"$regex": regex}},
                        {"skills": {"$elemMatch": {"$regex": regex}}},
                        {"role": {"$regex": regex}},
                    ]},
                    {"name": 1, "role": 1, "skills": 1, "total_experience": 1},
                ).limit(limit)
            )

    # If we have embeddings, re-rank by semantic similarity quickly
    if openai_client and docs:
        try:
            qemb = openai_client.embeddings.create(model=OPENAI_MODEL_EMBEDDING, input=query).data[0].embedding  # type: ignore
            ids = [d["_id"] for d in docs]
            full_docs = list(candidates_col.find({"_id": {"$in": ids}}, {VECTOR_FIELD: 1, "name": 1, "role": 1, "skills": 1, "total_experience": 1}))
            id_to_doc = {d["_id"]: d for d in full_docs}
            rescored = []
            for d in docs:
                v = id_to_doc.get(d["_id"], {}).get(VECTOR_FIELD)
                sim = cosine_similarity(qemb, v) if isinstance(v, list) else 0.0
                rescored.append((sim, d))
            rescored.sort(key=lambda x: x[0], reverse=True)
            docs = [d for _, d in rescored][:limit]
        except Exception as e:
            logger.debug("Shortlist semantic rerank failed: %s", e)

    return docs


def build_candidate_summary(doc: Dict[str, Any]) -> str:
    name = doc.get("name") or "Unknown"
    role = doc.get("role") or ""
    skills = ", ".join((doc.get("skills") or [])[:25])
    years = doc.get("total_experience", {}).get("years", 0)
    return f"Name: {name}; Role: {role}; Skills: {skills}; YearsExperience: {years}"


# ------------------------------------------------------------
# Error Handlers
# ------------------------------------------------------------


@app.errorhandler(413)
def request_entity_too_large(error):  # type: ignore
    return error_response("File too large.", 413)


@app.errorhandler(404)
def not_found(error):  # type: ignore
    return error_response("Not found", 404)


@app.errorhandler(Exception)
def handle_exception(e):  # type: ignore
    logger.exception("Unhandled error: %s", e)
    return error_response("Internal server error", 500)


# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------


@app.route("/health", methods=["GET"]) 
def health() -> Response:
    """Health check endpoint."""
    return json_response({"status": "ok"})


@app.route("/upload", methods=["POST"]) 
def upload_resume() -> Response:
    """Upload a resume (PDF/DOC/DOCX), parse into structured profile, and store.

    - Accepts multipart/form-data with field `file`.
    - Extracts raw text with pdfminer/python-docx/textract.
    - Uses GPT to parse into structured JSON (skills enforced as list of strings).
    - Stores profile, raw text, and schedules background embedding computation.
    - Upserts by `email` when available; otherwise creates a new candidate.
    """
    if "file" not in request.files:
        return error_response("Missing file field")
    file = request.files["file"]
    if not file or not file.filename:
        return error_response("No file uploaded")

    try:
        raw_text = extract_text_from_file(file)
        if not raw_text:
            return error_response("Failed to extract text from file", 400)
    except Exception as e:
        return error_response("Text extraction failed", 400, {"reason": str(e)})

    try:
        profile = parse_resume_with_gpt(raw_text)
    except Exception as e:
        return error_response("Resume parsing failed", 502, {"reason": str(e)})

    now = datetime.utcnow()
    doc = {
        **profile,
        "raw_text": raw_text,
        "created_at": now,
        "updated_at": now,
    }

    # Upsert by email if available
    email = profile.get("email")
    try:
        if email:
            existing = candidates_col.find_one({"email": email})
            if existing:
                candidates_col.update_one({"_id": existing["_id"]}, {"$set": doc})
                candidate_id = existing["_id"]
            else:
                res = candidates_col.insert_one(doc)
                candidate_id = res.inserted_id
        else:
            res = candidates_col.insert_one(doc)
            candidate_id = res.inserted_id
    except DuplicateKeyError:
        # In race, fetch and update
        existing = candidates_col.find_one({"email": email})
        if existing:
            candidates_col.update_one({"_id": existing["_id"]}, {"$set": doc})
            candidate_id = existing["_id"]
        else:
            res = candidates_col.insert_one(doc)
            candidate_id = res.inserted_id
    except Exception as e:
        logger.error("DB error on upload: %s", e)
        return error_response("Database error", 500)

    try:
        schedule_embedding(profile, raw_text, candidate_id)
    except Exception as e:
        logger.warning("Failed to schedule embedding: %s", e)

    response_payload = {
        "id": str(candidate_id),
        "profile": profile,
        "message": "Uploaded and parsed. Embedding scheduled.",
    }
    return json_response(response_payload, 201)


@app.route("/search", methods=["GET", "POST"]) 
def search_route() -> Response:
    """Keyword and fuzzy search across candidates.

    Query params or JSON body:
    - q: search query (string)
    - limit: optional int (default 20)
    Returns list of candidates with confidence score [0,1].
    """
    data = request.get_json(silent=True) or {}
    q = request.args.get("q") or data.get("q")
    if not q or not isinstance(q, str):
        return error_response("Missing query 'q'")
    try:
        limit = int(request.args.get("limit") or data.get("limit") or 20)
    except Exception:
        limit = 20

    try:
        # Prefer Mongo text search
        cursor = candidates_col.find(
            {"$text": {"$search": q}},
            {"score": {"$meta": "textScore"}, "name": 1, "email": 1, "role": 1, "skills": 1, "location": 1, "total_experience": 1},
        ).sort([("score", {"$meta": "textScore"})]).limit(limit)
        results = []
        max_score = None
        items = list(cursor)
        for d in items:
            s = d.get("score", 1.0)
            if max_score is None or s > max_score:
                max_score = s
        max_score = max_score or 1.0
        for d in items:
            results.append({
                "id": str(d["_id"]),
                "name": d.get("name"),
                "email": d.get("email"),
                "role": d.get("role"),
                "skills": d.get("skills", []),
                "location": d.get("location"),
                "total_experience": d.get("total_experience", {}),
                "confidence": float(d.get("score", 0.0)) / float(max_score or 1.0),
            })
        return json_response({"query": q, "results": results})
    except Exception as e:
        logger.debug("$text search failed, falling back to regex: %s", e)

    # Fallback: regex match + naive scoring
    tokens = [t.lower() for t in re.findall(r"\w+", q)]
    regex = re.compile("|".join([re.escape(t) for t in tokens]), re.IGNORECASE) if tokens else None
    matches = []
    if regex:
        for d in candidates_col.find({}, {"name": 1, "email": 1, "role": 1, "skills": 1, "location": 1, "total_experience": 1, "raw_text": 1}).limit(1000):
            haystack = " ".join([
                d.get("name", ""), d.get("role", ""), " ".join(d.get("skills", [])), d.get("raw_text", "")
            ]).lower()
            hits = len(regex.findall(haystack))
            if hits > 0:
                matches.append((hits, d))
    matches.sort(key=lambda x: x[0], reverse=True)
    matches = matches[:limit]
    max_hits = matches[0][0] if matches else 1
    results = [{
        "id": str(d["_id"]),
        "name": d.get("name"),
        "email": d.get("email"),
        "role": d.get("role"),
        "skills": d.get("skills", []),
        "location": d.get("location"),
        "total_experience": d.get("total_experience", {}),
        "confidence": float(h) / float(max_hits or 1),
    } for h, d in matches]
    return json_response({"query": q, "results": results})


@app.route("/semantic-search", methods=["GET", "POST"]) 
def semantic_search_route() -> Response:
    """Vector-based semantic search using cosine similarity over embeddings.

    Query params or JSON body:
    - q: search query (string)
    - threshold: optional float [0,1], default 0.75
    - limit: optional int, default 20
    Returns list of candidates with confidence score [0,1] (cosine similarity).
    """
    if not openai_client:
        return error_response("OpenAI not configured for embeddings", 503)

    data = request.get_json(silent=True) or {}
    q = request.args.get("q") or data.get("q")
    if not q or not isinstance(q, str):
        return error_response("Missing query 'q'")
    try:
        threshold = float(request.args.get("threshold") or data.get("threshold") or os.getenv("SEMANTIC_THRESHOLD", "0.75"))
    except Exception:
        threshold = 0.75
    try:
        limit = int(request.args.get("limit") or data.get("limit") or 20)
    except Exception:
        limit = 20

    try:
        qemb = openai_client.embeddings.create(model=OPENAI_MODEL_EMBEDDING, input=q).data[0].embedding  # type: ignore
    except Exception as e:
        return error_response("Failed to compute query embedding", 502, {"reason": str(e)})

    docs = list(candidates_col.find({VECTOR_FIELD: {"$type": "array"}}, {"name": 1, "email": 1, "role": 1, "skills": 1, "location": 1, "total_experience": 1, VECTOR_FIELD: 1}).limit(5000))
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for d in docs:
        sim = cosine_similarity(qemb, d.get(VECTOR_FIELD) or [])
        if sim >= threshold:
            scored.append((sim, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    results = [{
        "id": str(d["_id"]),
        "name": d.get("name"),
        "email": d.get("email"),
        "role": d.get("role"),
        "skills": d.get("skills", []),
        "location": d.get("location"),
        "total_experience": d.get("total_experience", {}),
        "confidence": float(s),
    } for s, d in scored[:limit]]
    return json_response({"query": q, "threshold": threshold, "results": results})


@app.route("/ai-job-match", methods=["GET", "POST"]) 
def ai_job_match() -> Response:
    """AI-driven job matching.

    Accepts job requirement text via query param `text` or JSON body {text}.
    Returns top 5 candidates with score 0–100 and reason, sorted by score.
    """
    if not openai_client:
        return error_response("OpenAI not configured", 503)

    data = request.get_json(silent=True) or {}
    text = request.args.get("text") or data.get("text")
    if not text or not isinstance(text, str):
        return error_response("Missing 'text' with job requirements")

    # Build shortlist to keep prompt small
    shortlist = shortlist_candidates_for_job_text(text, limit=20)
    if not shortlist:
        return json_response({"query": text, "results": []})

    candidates_list = [
        {
            "id": str(d["_id"]),
            "summary": build_candidate_summary(d),
        }
        for d in shortlist
    ]

    system = (
        "You are an expert technical recruiter. Score candidates 0-100 for the given job requirements. "
        "Provide a short reason. Be objective: prioritize matching skills, relevant experience, and role fit."
    )
    user = (
        "Job Requirements:\n" + text +
        "\n\nCandidates (format: id | summary):\n" +
        "\n".join([f"{c['id']} | {c['summary']}" for c in candidates_list]) +
        "\n\nReturn ONLY valid minified JSON array of up to 5 objects: [{id, score, reason}]."
    )

    try:
        chat = openai_client.chat.completions.create(
            model=OPENAI_MODEL_RANKING,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
        )
        content = chat.choices[0].message.content or "{}"
        parsed = json.loads(content)
        # Allow either array or object with results
        items = parsed if isinstance(parsed, list) else parsed.get("results") or parsed.get("matches") or []
        out = []
        for item in items:
            cid = str(item.get("id"))
            score = float(item.get("score", 0))
            reason = str(item.get("reason", ""))
            # Attach candidate basic info
            doc = candidates_col.find_one({"_id": ObjectId(cid)}, {"name": 1, "email": 1, "role": 1, "skills": 1, "location": 1, "total_experience": 1})
            if doc:
                out.append({
                    "id": cid,
                    "name": doc.get("name"),
                    "email": doc.get("email"),
                    "role": doc.get("role"),
                    "skills": doc.get("skills", []),
                    "location": doc.get("location"),
                    "total_experience": doc.get("total_experience", {}),
                    "score": max(0, min(100, round(score))),
                    "reason": reason,
                })
        out.sort(key=lambda x: x.get("score", 0), reverse=True)
        out = out[:5]
        return json_response({"query": text, "results": out})
    except Exception as e:
        logger.error("AI job match failed: %s", e)
        return error_response("AI job match failed", 502, {"reason": str(e)})


@app.route("/voice-job-match", methods=["POST"]) 
def voice_job_match() -> Response:
    """Voice-based job matching.

    - Accepts audio file (mp3, wav, m4a, ogg, etc) in multipart/form-data under `file` or `audio`.
    - Transcribes with Whisper.
    - Matches candidates as in /ai-job-match.
    - Returns multipart/mixed response with JSON results and an audio TTS summary of the top match.
    Optional params: top_n (default 1) to include more candidates in TTS summary.
    """
    if not openai_client:
        return error_response("OpenAI not configured", 503)

    f = request.files.get("audio") or request.files.get("file")
    if not f or not f.filename:
        return error_response("Missing audio file under 'audio' or 'file'")

    try:
        # Whisper transcription
        audio_bytes = read_file_bytes(f)
        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=(f.filename, io.BytesIO(audio_bytes)),
        )
        text = transcript.text  # type: ignore
        if not text:
            return error_response("Transcription failed", 502)
    except Exception as e:
        logger.error("Transcription failed: %s", e)
        return error_response("Transcription failed", 502, {"reason": str(e)})

    # Reuse AI job match to get results
    with app.test_request_context(json={"text": text}):
        json_result_resp = ai_job_match()
    if json_result_resp.status_code != 200:
        return json_result_resp

    results_json = json.loads(json_result_resp.get_data(as_text=True))
    results = results_json.get("results", [])

    try:
        top_n = int(request.args.get("top_n") or request.form.get("top_n") or 1)
    except Exception:
        top_n = 1
    top_n = max(1, min(5, top_n))

    # Build TTS summary
    if results:
        summaries = []
        for i, r in enumerate(results[:top_n]):
            summaries.append(
                f"Candidate {i+1}: {r.get('name') or 'Unknown'} — Score {r.get('score', 0)}. Reason: {r.get('reason', '')}"
            )
        tts_text = " ".join(summaries)
    else:
        tts_text = "No suitable candidates found."

    try:
        speech = openai_client.audio.speech.create(
            model=OPENAI_MODEL_TTS,
            voice="alloy",
            input=tts_text,
        )
        audio_bytes_tts = speech.read()  # type: ignore
    except Exception as e:
        logger.error("TTS failed: %s", e)
        # Return only JSON if TTS fails
        return json_response({"transcript": text, **results_json})

    # Multipart/mixed response
    boundary = f"BOUNDARY-{uuid.uuid4()}"
    body = []
    # Part 1: JSON
    body.append(f"--{boundary}\r\n")
    body.append("Content-Type: application/json\r\n\r\n")
    body.append(json.dumps({"transcript": text, **results_json}, separators=(",", ":")))
    body.append("\r\n")
    # Part 2: Audio (mp3)
    body.append(f"--{boundary}\r\n")
    body.append("Content-Type: audio/mpeg\r\n")
    body.append("Content-Disposition: attachment; filename=summary.mp3\r\n\r\n")
    body_bytes = "".join(body).encode("utf-8") + audio_bytes_tts + f"\r\n--{boundary}--\r\n".encode("utf-8")

    return Response(body_bytes, status=200, mimetype=f"multipart/mixed; boundary={boundary}")


# ------------------------------------------------------------
# Atlas Vector Search Placeholder
# ------------------------------------------------------------


def atlas_vector_search_placeholder(query_vector: List[float], threshold: float = 0.75, limit: int = 20) -> List[Dict[str, Any]]:
    """Placeholder for MongoDB Atlas Vector Search integration.

    To enable, set USE_ATLAS_VECTOR_SEARCH=true and implement an aggregation using
    $vectorSearch (or $knnBeta) targeting the `embedding` field.
    """
    if not USE_ATLAS_VECTOR_SEARCH:
        return []
    # Example sketch (to be implemented by enabling Atlas Vector Search):
    # pipeline = [
    #   {
    #       "$vectorSearch": {
    #           "index": "vector_index_name",
    #           "path": VECTOR_FIELD,
    #           "queryVector": query_vector,
    #           "numCandidates": 200,
    #           "limit": limit,
    #       }
    #   },
    #   {"$project": {"name": 1, "email": 1, "role": 1, "skills": 1, "location": 1, "total_experience": 1, "score": {"$meta": "vectorSearchScore"}}}
    # ]
    # return list(candidates_col.aggregate(pipeline))
    return []


# ------------------------------------------------------------
# Main entrypoint
# ------------------------------------------------------------


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    logger.info("Starting app on %s:%s (debug=%s)", host, port, debug)
    app.run(host=host, port=port, debug=debug)

