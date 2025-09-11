from flask import Flask, request, jsonify
import os, json, re
import google.generativeai as genai
from werkzeug.utils import secure_filename
from pdfminer.high_level import extract_text as extract_pdf_text
from dotenv import load_dotenv
import docx
from pymongo import MongoClient

from bson import ObjectId

# ==========================
# CONFIG
# ==========================
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
ALLOWED_EXT = {"pdf", "docx", "txt"}

# Gemini API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)

# ==========================
# MONGODB INIT
# ==========================
# Local MongoDB (default)
client = MongoClient("mongodb://localhost:27017/")
db = client["cv_analyzer"]
candidates_collection = db["candidates"]

# ==========================
# FILE TEXT EXTRACTION
# ==========================
def extract_text(path):
    ext = path.split(".")[-1].lower()
    if ext == "pdf":
        return extract_pdf_text(path)
    elif ext == "docx":
        doc = docx.Document(path)
        return "\n".join([p.text for p in doc.paragraphs])
    elif ext == "txt":
        return open(path, "r", encoding="utf-8", errors="ignore").read()
    return ""

# ==========================
# GEMINI PARSING
# ==========================
PROMPT = """
Extract the following details from the resume and return JSON and return year in jason array in objects where needed ans dont add year where is not metioned:
{{
  "name": "",
  "email": "",
  "phone": "",
  "location": "",
  "education": [],
  "experience": [],
  "skills": []
}}
Resume:
{resume}
"""




def parse_resume(resume_text):
    model = genai.GenerativeModel("gemini-1.5-flash")

    response = model.generate_content(
        PROMPT.format(resume=resume_text[:20000]),
        generation_config=genai.types.GenerationConfig(
            response_mime_type="application/json"
        )
    )

    try:
        return json.loads(response.text)
    except Exception as e:
        print("JSON parse error:", e, "RAW:", response.text)
        return {
            "name": None,
            "email": None,
            "phone": None,
            "skills": [],
            "experience": [],
            "education": [],
            "location": None
        }

# ==========================
# ROUTES
# ==========================
@app.route("/upload", methods=["POST"])
def upload_resume():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files["file"]
    filename = secure_filename(file.filename)
    if not filename or filename.split(".")[-1].lower() not in ALLOWED_EXT:
        return jsonify({"error": "Invalid file type"}), 400

    path = os.path.join(UPLOAD_DIR, filename)
    file.save(path)

    text = extract_text(path)

    # 👇 Debug: check what text we got
    print("=== Extracted Resume Text ===")
    print(text[:1000])  # only first 1000 chars so console doesn’t explode

    from datetime import datetime

    def calculate_total_experience(experiences):
        """
        experiences: list of dicts like:
        [
          {"title": "Software Engineer", "company": "ABC", "start": "Jan 2020", "end": "Dec 2022"},
          {"title": "Senior Engineer", "company": "XYZ", "start": "Jan 2023", "end": None}
        ]
        """
        total_months = 0
        for exp in experiences:
            start_str = exp.get("start")
            end_str = exp.get("end")

            if not start_str:
                continue

            try:
                start = datetime.strptime(start_str, "%b %Y")  # e.g. Jan 2020
            except:
                try:
                    start = datetime.strptime(start_str, "%Y")  # fallback: 2020
                except:
                    continue

            if end_str:
                try:
                    end = datetime.strptime(end_str, "%b %Y")
                except:
                    try:
                        end = datetime.strptime(end_str, "%Y")
                    except:
                        end = datetime.today()
            else:
                end = datetime.today()

            total_months += (end.year - start.year) * 12 + (end.month - start.month)

        total_years = round(total_months / 12, 2)  # keep 2 decimals
        return total_years

    parsed = parse_resume(text)

    # ==========================
    # DUPLICATE CHECK
    # ==========================
    # First check by raw_text
    existing = candidates_collection.find_one({"raw_text": text})

    # If email exists, check that too (more reliable than raw text)
    if not existing and parsed.get("email"):
        existing = candidates_collection.find_one({"email": parsed.get("email")})

    if existing:
        return jsonify({
            "ok": False,
            "message": "This CV is already in the database",
            "id": str(existing["_id"]),
            "parsed": {
                "name": existing.get("name"),
                "email": existing.get("email"),
                "phone": existing.get("phone"),
                "skills": existing.get("skills"),
                "experience": existing.get("experience"),
                "education": existing.get("education"),
                "location": existing.get("location")
            }
        }), 409

    # Insert into MongoDB
    candidate_doc = {
        "name": parsed.get("name"),
        "email": parsed.get("email"),
        "phone": parsed.get("phone"),
        "skills": parsed.get("skills") or [],
        "experience": parsed.get("experience") or [],
        "education": parsed.get("education") or [],
        "location": parsed.get("location"),
        "raw_text": text
    }
    inserted = candidates_collection.insert_one(candidate_doc)

    return jsonify({"ok": True, "id": str(inserted.inserted_id), "parsed": parsed})


@app.route("/search")
def search():
    skill_query = request.args.get("skill", "")
    name_query = request.args.get("name", "")
    email_query = request.args.get("email", "")

    results = []

    query_parts = []

    # Skill search (matches any skill in comma-separated list)
    if skill_query:
        skills = [s.strip() for s in skill_query.split(",")]
        query_parts.append({"$or": [{"skills": {"$regex": s, "$options": "i"}} for s in skills]})

    # Name search (partial match)
    if name_query:
        query_parts.append({"name": {"$regex": name_query, "$options": "i"}})

    # Email search (partial match)
    if email_query:
        query_parts.append({"email": {"$regex": email_query, "$options": "i"}})



    # Combine query parts
    if query_parts:
        query = {"$and": query_parts}  # all conditions must match
    else:
        query = {}  # if no filters, return all

    cursor = candidates_collection.find(query)

    for doc in cursor:
        results.append({
            "id": str(doc["_id"]),
            "name": doc.get("name"),
            "email": doc.get("email"),
            "phone": doc.get("phone"),
            "skills": doc.get("skills", []),
            "experience": doc.get("experience", []),
            "education": doc.get("education", []),
            "location": doc.get("location"),
            "raw_text": doc.get("raw_text")
        })
    if not results:
        return jsonify({"ok": False, "message": "No data to show"}), 404

    return jsonify({"count": len(results), "results": results})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
