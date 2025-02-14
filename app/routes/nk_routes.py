import os
from flask import Blueprint, request, jsonify, session, send_file, current_app
from ..const import FileDir, FrontEndWarning, Consts
import traceback
from ..user import User
from utils import session_manage
from ..device import DeviceStructure
import logging


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


nk_bp = Blueprint('nk_bp', __name__)


@nk_bp.route('/set_nk', methods=['POST'])
def set_nk():
    try:
        rfi_source = request.form.get("rfi_source")
        layer_order = int(request.form.get("layer_order"))
        std_material_name = request.form.get("standard_filename")
        user = User(session.get("Email"))

        device: DeviceStructure = session_manage.load_device_from_session(session)  # Load from session
        logger.info("上传前器件结构：")
        print(device)

        if rfi_source == Consts.NK_TYPE_CONST:     # 常数折射率
            rfi_real = float(request.form.get("real"))
            rfi_imag = float(request.form.get("imag"))
            device.get_layer(layer_order).set_nk(rfi_source, rfi_real=rfi_real, rfi_imag=rfi_imag)
        elif rfi_source == Consts.NK_TYPE_SYSTEM:  # 系统折射率
            device.get_layer(layer_order).set_nk(rfi_source, std_material_name=std_material_name)
        elif rfi_source == Consts.NK_TYPE_CUSTOM:  # 自定义折射率上传
            posted_file = request.files["rfi_file"]  # FileStorage object

            if posted_file is None:
                return jsonify({"error": FrontEndWarning.FILE_ERROR}), 400
            if posted_file.filename == '':
                return jsonify({'error': 'No selected file'}), 400

            rfi_file_path = os.path.join(FileDir.TMP_DIR, posted_file.filename)
            posted_file.save(os.path.join(FileDir.TMP_DIR, posted_file.filename))
            if device.get_layer(layer_order).set_nk(rfi_source, rfi_file=rfi_file_path, user=user):
                logger.info("上传后器件结构：")
                print(device)
                session_manage.save_device_to_session(session, device)  # Save to session

                return jsonify({"message": "Custom refractive index data uploaded successfully",
                                "rfi_path": device.get_layer(layer_order).get_nk().get_rfi_file_path()}), 200
        else:
            return jsonify({"error": FrontEndWarning.INVALID_SOURCE}), 400

        logger.info("上传后Session信息：")

        session_manage.save_device_to_session(session, device)  # Save to session
        logger.info(f"当前用户：{session.get('Email')}")
        logger.info(f"当前用户的器件结构为：")
        print(device)


        return jsonify({"message": "Refractive index data located successfully",
                        "rfi_path": device.get_layer(layer_order).get_nk().get_rfi_file_path()}), 200
    except FileNotFoundError as fnf:
        traceback.print_exc()
        return jsonify({"error": str(fnf), "message": "系统中不存在该折射率文件，请上传"}), 404
    except KeyError as ke:
        traceback.print_exc()
        return jsonify({"error": f"Missing key in Form-data: {ke}"}), 400


@nk_bp.route("/get_sys_nk", methods=["GET"])
def get_system_nk_file():
    try:
        material = request.args.get("material")
        target_file = os.path.join(os.path.dirname(current_app.root_path), "refractive_index_csv", "system", f"{material}.csv")
        if not os.path.exists(target_file):
            raise FileNotFoundError("File not exists")
        return send_file(target_file, as_attachment=True)
    except FileNotFoundError as fnf:
        traceback.print_exc()
        return jsonify({"error": str(fnf), "message": "系统中不存在该折射率文件，请上传"}), 404
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@nk_bp.route("/system_files", methods=["GET"])
def get_system_nk_files():
    try:
        files = sorted(os.listdir(os.path.join("refractive_index_csv", "system")))
        materials = []
        for file in files:
            materials.append(file.split(".")[0])
        return jsonify(materials), 200
    except FileNotFoundError as fnf:
        traceback.print_exc()
        return jsonify({"error": str(fnf), "message": "系统折射率目录不存在！"}), 404
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
