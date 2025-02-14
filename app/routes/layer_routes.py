from flask import Blueprint, jsonify, request
from ..device import Layer

layer_bp = Blueprint('layer_bp', __name__)


@layer_bp.route('/layer', methods=['POST'])
def add_layer():
    data = request.json  # POST 方法使用body传递参数，用此方法获取
    print(data)
    try:
        layer = Layer(data["layer_Id"], data['material'], data['color'], int(data['thickness']), data['layer_type'])
        return jsonify({"message": "Layer added successfully"}), 201
    except (AssertionError, KeyError, TypeError) as e:
        return jsonify({"error": str(e)}), 400


@layer_bp.route('/layers/<int:layer_id>', methods=['DELETE'])
def delete_layer(layer_id):
    try:
        return jsonify({"message": "Layer removed successfully"}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@layer_bp.route('/layers/<int:layer_id>/up', methods=['PUT'])
def layer_up(layer_id):
    """Move a layer up, return the new layer order"""
    try:
        return jsonify({"message": "Layer moved up successfully"}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@layer_bp.route('/layers/<int:layer_id>/down', methods=['PUT'])
def layer_down(layer_id):
    """Move a layer down, return the new layer order"""
    try:
        return jsonify({"message": "Layer moved down successfully"}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
