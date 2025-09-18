import os
from flask import Flask, send_from_directory
from .db import init_mongo
from .embeddings import init_embeddings
from .routes import api_bp
from config import get_config


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", static_url_path="/static")
    app.config.from_object(get_config())

    # Ensure upload folder exists
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # Initialize MongoDB client and store on app
    init_mongo(app)

    # Initialize embeddings provider and store on app
    init_embeddings(app)

    # Register API blueprint
    app.register_blueprint(api_bp, url_prefix="/api")

    # Frontend: serve index.html
    @app.route("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    return app