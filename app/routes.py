from __future__ import annotations
import os
import math
import uuid
from typing import Any, Dict, List
from flask import Blueprint, current_app, jsonify, request
from werkzeug.utils import secure_filename
from .db import get_db, now_utc
from .parsers import read_cv_text, extract_metadata
from .embeddings import get_embedder


api_bp = Blueprint("api", __name__)

ALLOWED_EXT = {"pdf", "docx", "txt"}


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for va, vb in zip(a, b):
        dot += va * vb
        na += va * va
        nb += vb * vb
    denom = math.sqrt(na) * math.sqrt(nb)
    return (dot / denom) if denom else 0.0


@api_bp.route("/upload", methods=["POST"])
def upload_cv():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Unsupported file type: .{ext}"}), 400

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    save_path = os.path.join(current_app.config["UPLOAD_FOLDER"], unique_name)
    file.save(save_path)

    text, detected_type = read_cv_text(save_path)
    meta = extract_metadata(text)

    embedder = get_embedder()
    embedding = embedder.embed(text)

    doc = {
        "file_name": filename,
        "stored_name": unique_name,
        "file_type": detected_type,
        "raw_text": text,
        **meta,
        "embedding": embedding,
        "created_at": now_utc(),
    }

    db = get_db()
    result = db.cvs.insert_one(doc)
    doc_id = str(result.inserted_id)

    response = {"id": doc_id, **{k: doc[k] for k in ["name", "email", "phone", "skills", "total_experience_years", "file_name", "file_type"]}}
    return jsonify(response), 201


@api_bp.route("/search", methods=["GET"])
def search_cvs():
    db = get_db()
    name = request.args.get("name")
    email = request.args.get("email")
    text = request.args.get("text")
    min_exp = request.args.get("min_experience")

    query: Dict[str, Any] = {}
    if name:
        query["name"] = {"$regex": name, "$options": "i"}
    if email:
        query["email"] = {"$regex": email, "$options": "i"}
    if text:
        query["raw_text"] = {"$regex": text, "$options": "i"}
    if min_exp:
        try:
            query["total_experience_years"] = {"$gte": float(min_exp)}
        except ValueError:
            pass

    cursor = db.cvs.find(query).limit(100)
    results: List[Dict[str, Any]] = []
    for d in cursor:
        d["id"] = str(d.pop("_id"))
        d.pop("embedding", None)
        d.pop("raw_text", None)
        results.append(d)

    return jsonify({"results": results})


def rank_candidates(query_text: str, limit: int = 5) -> List[Dict[str, Any]]:
    db = get_db()
    embedder = get_embedder()
    qvec = embedder.embed(query_text)

    docs = list(db.cvs.find({}, {"embedding": 1, "name": 1, "email": 1, "skills": 1, "total_experience_years": 1}).limit(1000))
    scored: List[Dict[str, Any]] = []
    lower_query = (query_text or "").lower()
    for d in docs:
        emb = d.get("embedding")
        if not emb:
            continue
        score = cosine_similarity(qvec, emb)
        skill_overlap = 0
        if d.get("skills"):
            for s in d["skills"]:
                if s in lower_query:
                    skill_overlap += 1
        exp = float(d.get("total_experience_years") or 0.0)
        boosted = score + 0.02 * exp + 0.05 * min(skill_overlap, 5)
        d_copy = {
            "id": str(d.get("_id")),
            "name": d.get("name"),
            "email": d.get("email"),
            "skills": d.get("skills", []),
            "total_experience_years": exp,
            "score": float(boosted),
        }
        scored.append(d_copy)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:limit]


@api_bp.route("/ai/search", methods=["POST"])
def ai_search():
    data = request.get_json(silent=True) or {}
    query_text = data.get("query", "").strip()
    top_k = int(data.get("top_k", 5))
    if not query_text:
        return jsonify({"error": "Missing query"}), 400

    top = rank_candidates(query_text, limit=top_k)

    reasons: List[Dict[str, Any]] = []
    for idx, c in enumerate(top, start=1):
        skill_list = ", ".join(c.get("skills", [])[:10])
        reasons.append({
            "id": c["id"],
            "reason": f"Rank {idx}: match score {c['score']:.3f}. Skills: {skill_list}. Experience: {c['total_experience_years']} years.",
        })

    return jsonify({"results": top, "reasons": reasons})


@api_bp.route("/chat", methods=["POST"])
def chat_ai():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "Missing message"}), 400

    top = rank_candidates(message, limit=5)
    if not top:
        return jsonify({"reply": "I couldn't find matching CVs yet. Try uploading some resumes first.", "results": []})

    lines = []
    for c in top:
        lines.append(f"- {c.get('name') or 'Unknown'} ({c.get('total_experience_years', 0)} yrs): skills {', '.join(c.get('skills', [])[:8])}")
    reply = "Here are the top candidates I found:\n" + "\n".join(lines)

    return jsonify({"reply": reply, "results": top})