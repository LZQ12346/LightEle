import os
import time
import traceback
from io import BytesIO
from ..device import DeviceStructure
from ..light import Light
from ..task import Task
from ..user import User
from flask import Blueprint, jsonify, send_file, session
from ..const import FileDir, FileName
from utils import session_manage
import logging


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


cal_bp = Blueprint('cal_bp', __name__)


@cal_bp.route('/lightele', methods=['GET', 'POST'])
def calculate():
    try:
        lam: Light = session_manage.load_light_from_session(session)
        device: DeviceStructure = session_manage.load_device_from_session(session)
        user: User = User(session.get("Email"))
        user_id: int = user.get_userId()
        logger.info(f"当前用户ID为 {user_id}")
        logger.info(f"当前用户邮箱为 {session.get('Email')}")
        logger.info(f"当前用户的光源信息为: {lam}")
        logger.info(f"当前用户的器件结构为: ")
        print(device)

        if not device or not lam:
            print("No device or light data")
            return jsonify({"error": "No device or light data"}), 400
        task = Task(str(user_id), device, lam)       # nk 没有设置，导致此行出问题
        session["time"] = task.run_time
        logger.info(f"当前任务的时间戳为 {task.run_time}")
        task.run()                                   # 材料厚度过大，非常耗时
        while not task.check_result_files():         # 增加同步机制。由于文件写入磁盘是异步的，所以需要等待文件写入完成
            time.sleep(0.5)
        logger.info("文件已全部写入完成！")
        return jsonify({"message": "success"}), 200  # 必须等到计算完成并且结果文件生成后才能返回成功
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@cal_bp.route('/download/', methods=['GET'])
def download():
    user = User(session.get("Email"))
    zip_file_dir = os.path.join(FileDir.DOWNLOAD_DIR, str(user.get_userId()))
    zip_file = sorted(os.listdir(zip_file_dir))[-1]
    return send_file(os.path.join("..", zip_file_dir, zip_file), as_attachment=True)


@cal_bp.route("/show_art", methods=['GET'])
def get_img1():
    user = User(session.get("Email"))
    image_path = os.path.join(FileDir.IMAGES_DIR, str(user.get_userId()), session["time"], FileName.ART_IMG)

    with open(image_path, 'rb') as img_file:
        img = BytesIO(img_file.read())
        img.seek(0)

    return send_file(img, mimetype='image/jpg')


@cal_bp.route('/show_img2', methods=['GET'])
def get_img2():
    user = User(session.get("Email"))
    image_path = os.path.join(FileDir.IMAGES_DIR, str(user.get_userId()), session["time"], FileName.Gx_IMG)

    with open(image_path, 'rb') as img_file:
        img = BytesIO(img_file.read())
        img.seek(0)

    return send_file(img, mimetype='image/jpg')

@cal_bp.route('/show_img3', methods=['GET'])
def get_img3():
    user = User(session.get("Email"))
    image_dir = os.path.join(FileDir.IMAGES_DIR, str(user.get_userId()), session["time"])
    E_jpgs = []
    for filename in os.listdir(image_dir):
        if filename.startswith(FileName.E_FIELD_IMG_PREFIX) and filename.endswith(".jpg"):
            E_jpgs.append(filename)
    E_jpgs.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    with open(os.path.join(image_dir, E_jpgs[-1]), 'rb') as img_file:
        img = BytesIO(img_file.read())
        img.seek(0)

    return send_file(img, mimetype='image/png')


@cal_bp.route('/show_absorption', methods=['GET'])  # Deprecated
def get_absorption():
    image_path = os.path.join(FileDir.IMAGES_DIR, sorted(os.listdir(FileDir.IMAGES_DIR))[0])

    with open(image_path, 'rb') as img_file:
        img = BytesIO(img_file.read())
        img.seek(0)

    return send_file(img, mimetype='image/png')


@cal_bp.route('/show_reflection', methods=['GET'])  # Deprecated
def get_reflection():
    image_path = os.path.join(FileDir.IMAGES_DIR, sorted(os.listdir(FileDir.IMAGES_DIR))[1])

    with open(image_path, 'rb') as img_file:
        img = BytesIO(img_file.read())
        img.seek(0)

    return send_file(img, mimetype='image/png')


@cal_bp.route('/show_transmission', methods=['GET'])  # Deprecated
def get_transmission():
    image_path = os.path.join(FileDir.IMAGES_DIR, sorted(os.listdir(FileDir.IMAGES_DIR))[2])

    with open(image_path, 'rb') as img_file:
        img = BytesIO(img_file.read())
        img.seek(0)

    return send_file(img, mimetype='image/png')
