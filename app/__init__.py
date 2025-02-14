from flask import Flask, request
from .config import Config
from .models import init_db


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    init_db()
    
    from .routes import main_bp
    app.register_blueprint(main_bp)
    return app
