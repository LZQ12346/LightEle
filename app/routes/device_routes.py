import traceback
from flask import Blueprint, jsonify, request, session
from ..device import DeviceStructure, Layer
from utils import session_manage
import logging


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

device_bp = Blueprint('device_bp', __name__)


@device_bp.route('/device', methods=['GET'])
def get_device():
    """获取已创建的器件结构"""
    try:
        device = session_manage.load_device_from_session(session)
        if device.is_empty_device():
            return jsonify({"warning": "目前暂无添加层结构！"}), 200
        return jsonify(device.to_dict()), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@device_bp.route("/upload_device", methods=["POST"])
def upload_device_data():
    if not request.is_json:
        return jsonify({"msg": "Missing JSON in request"}), 400

    data = request.get_json()
    try:
        device_structure = DeviceStructure()
        layers = []
        for layer_data in data:
            layer = Layer(
                layer_data['layer_order'],
                layer_data['material'],
                layer_data['color'],
                layer_data['thickness'],
                layer_data['layer_type'],
                None
            )
            layers.append(layer)
        device_structure.set_layers(layers)
        session_manage.save_device_to_session(session, device_structure)  # Save to session

        logger.info(f"当前用户{session['Email']}上传了器件结构数据")
        logger.info(f"器件结构数据：")
        for layer in session['device_structure']:
            print(layer)

        return jsonify({"message": "Data uploaded successfully",
                        "layers": [repr(layer) for layer in layers]}), 200
    except KeyError as e:
        return jsonify({"msg": f"Missing key in JSON data: {e}"}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"msg": str(e)}), 500
