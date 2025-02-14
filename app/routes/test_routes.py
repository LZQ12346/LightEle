from flask import Blueprint, jsonify, request, session
from utils.session_manage import load_device_from_session

test_bp = Blueprint('test_bp', __name__)


@test_bp.route('/')
def home():
    return jsonify({"message": "Welcome to the Flask App!"})


@test_bp.route('/session', methods=['GET'])
def test():
    """Check session content"""
    if email := session.get("Email"):
        print("Email:", email)
    if device_structure := session.get("device_structure"):
        print("device_structure:")
        device = load_device_from_session(session)
        print(device)

    return jsonify({"session": dict(session)}), 200
