from flask import Blueprint, request, jsonify, session
from utils import session_manage
from ..light import Light
import logging


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

light_bp = Blueprint('light_bp', __name__)


@light_bp.route('/set', methods=['POST'])
def set_light():
    data = request.get_json()
    try:
        lam = Light(data['min_lam'], data['max_lam'], data['step_length'], data['CalAcc'], data['LightSelection'])
        if not lam.check_valid():
            raise ValueError('Invalid input')
        session_manage.save_light_to_session(session, lam)  # save to session
        logger.info(f"设定的光源信息为 {session.get('light')}")
        return jsonify({'message': 'success'}), 200
    except ValueError as ve:
        return jsonify({
            'message': 'failed',
            'error': str(ve)
        }), 400
    except Exception as e:
        return jsonify({
            'message': 'failed',
            'error': str(e)
        }), 500


@light_bp.route('/get_infor', methods=['GET'])
def get_light_infor():
    try:
        return jsonify(session.get("light")), 200
    except Exception as e:
        return jsonify({
            'message': 'failed',
            'error': str(e)
        }), 500


@light_bp.route('/get_lams', methods=['GET'])
def get_lams():
    try:
        lam = session_manage.load_light_from_session(session)
        if lam:
            return jsonify({'lams': lam.get_lams_in_list()}), 200
        else:
            return jsonify({'message': 'failed'}), 400
    except Exception as e:
        return jsonify({
            'message': 'failed',
            'error': str(e)
        }), 500
