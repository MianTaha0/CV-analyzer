import os
from dotenv import load_dotenv

load_dotenv()


class Config:
	SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "change-me")
	MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
	MONGO_DB = os.environ.get("MONGO_DB", "cvdb")
	UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "/workspace/uploads")

	EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "local")
	OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

	MAX_CONTENT_LENGTH = 20 * 1024 * 1024  # 20MB


def get_config() -> 'Config':
	return Config()