from __future__ import annotations
import datetime
from typing import Any
from flask import current_app
from pymongo import MongoClient, ASCENDING, TEXT


_mongo_client: MongoClient | None = None


def init_mongo(app: Any) -> None:
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(app.config["MONGO_URI"], tz_aware=True, serverSelectionTimeoutMS=1000)

    db = _mongo_client[app.config["MONGO_DB"]]
    cvs = db["cvs"]
    try:
        cvs.create_index([("email", ASCENDING)])
        cvs.create_index([("name", ASCENDING)])
        cvs.create_index([("total_experience_years", ASCENDING)])
        try:
            cvs.create_index([("raw_text", TEXT)], default_language="english")
        except Exception:
            pass
    except Exception:
        # Mongo may not be up yet; allow app to start
        pass


def get_db():
    assert _mongo_client is not None, "Mongo client not initialized"
    return _mongo_client[current_app.config["MONGO_DB"]]


def now_utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)