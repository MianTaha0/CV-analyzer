from __future__ import annotations
import os
import re
from typing import Dict, List, Tuple


def extract_text_from_pdf(file_path: str) -> str:
    from pdfminer.high_level import extract_text
    try:
        return extract_text(file_path) or ""
    except Exception:
        return ""


def extract_text_from_docx(file_path: str) -> str:
    from docx import Document
    try:
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        return ""


def extract_text_generic(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


COMMON_SKILLS = [
    "python", "java", "javascript", "node", "react", "angular", "c++", "c#", "go", "rust",
    "django", "flask", "fastapi", "pandas", "numpy", "scikit-learn", "ml", "machine learning",
    "data science", "sql", "mongodb", "postgres", "mysql", "aws", "azure", "gcp", "docker",
    "kubernetes", "linux", "git", "pytest", "selenium"
]


def extract_metadata(cv_text: str) -> Dict:
    text = cv_text or ""

    email_matches = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    email = email_matches[0] if email_matches else None

    phone_matches = re.findall(r"(?:\+\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3}[\s-]?\d{3,4}", text)
    phone = phone_matches[0] if phone_matches else None

    # Naive name extraction: first non-empty line that contains a space and no digits
    name = None
    for line in text.splitlines():
        line = line.strip()
        if len(line) > 2 and " " in line and not re.search(r"\d", line):
            name = line[:120]
            break

    skills_found: List[str] = []
    lower = text.lower()
    for skill in COMMON_SKILLS:
        if skill in lower:
            skills_found.append(skill)

    # Experience extraction: find the max years mentioned
    exp_years = 0.0
    for match in re.findall(r"(\d{1,2}(?:\.\d)?)\s*(?:\+?\s*)?(?:years|yrs|year)", lower):
        try:
            exp_years = max(exp_years, float(match))
        except ValueError:
            pass

    return {
        "name": name,
        "email": email,
        "phone": phone,
        "skills": sorted(set(skills_found)),
        "total_experience_years": exp_years,
    }


def read_cv_text(file_path: str) -> Tuple[str, str]:
    _, ext = os.path.splitext(file_path.lower())
    if ext in [".pdf"]:
        return extract_text_from_pdf(file_path), "pdf"
    if ext in [".docx"]:
        return extract_text_from_docx(file_path), "docx"
    return extract_text_generic(file_path), ext.lstrip(".") or "txt"