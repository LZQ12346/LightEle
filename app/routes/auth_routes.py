import re
from flask import Blueprint, request, session, jsonify, make_response
from ..models import register_db, get_user, store_verification_code, check_expiration, delete_code  # 假设这些函数在models模块中定义
from utils import send_verification_email,generate_verification_code
import logging


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


auth_bp = Blueprint('auth', __name__)

# 先查询用户是否存在未过期验证码。如果无才发送新验证码
@auth_bp.route('/send_verification_code', methods=['POST'])
def send_verification_code():
    data = request.json
    if data is None:
        return jsonify({'status': 'error', 'message': 'request is none'}), 400
    else:
        Email = data.get('Email')
        if not Email or Email == '':
            return jsonify({'status': 'error', 'message': 'Email is required'}), 400
        else:
            email_regex_1 = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
            email_regex_2 = r'\b[a-zA-Z0-9._%+-]+@(gmail\.com|yahoo\.com|outlook\.com|hotmail\.com|icloud\.com|aol\.com|qq\.com|163\.com|126\.com|yandex\.ru|mail\.com)\b'
            if not re.match(email_regex_1, Email):
                return jsonify({'status': 'error', 'message': 'incorrect email format'}), 400
            if re.match(email_regex_2, Email):
                return jsonify({'status': 'error', 'message': 'not recommend email addresses'}), 400

            if check_expiration(email=Email)==None:
                code = generate_verification_code()
                if send_verification_email(Email, code):
                    store_verification_code(Email, code)
                    return jsonify({'status': 'success', 'message': ''}), 200
                else:
                    return jsonify({'status': 'error', 'message': 'Failed to send verification code'}), 500 #  服务端出错返回500
            else:
                return jsonify({'status': 'error', 'message': 'code already sent less than 1 minute'}), 400


@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.json
    if data is None:
        return jsonify({'status': 'error', 'errors': [{'field': 'Request_Error', 'message': '请求体为空或格式不正确'}]}), 400
    else:
        Email = data['Email']
        EnterpriseOrUnit = data['EnterpriseOrUnit']
        UserPosition = data['UserPosition']
        UnitKind = data['UnitKind']
        code = data['code']

        email_regex_1 = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
        email_regex_2 = r'\b[a-zA-Z0-9._%+-]+@(gmail\.com|yahoo\.com|outlook\.com|hotmail\.com|icloud\.com|aol\.com|qq\.com|163\.com|126\.com|yandex\.ru|mail\.com)\b'

        errors = []
        if not re.match(email_regex_1, Email):
            errors.append({'field': 'Email_NameError', 'message': '请输入正确的邮箱格式'})
        if re.match(email_regex_2, Email):
            errors.append({'field': 'Email_TypeError', 'message': '请勿使用qq, 163, gmail邮箱，推荐使用教育/企业邮箱'})
        if len(EnterpriseOrUnit) >= 100:
            errors.append({'field': 'EnterpriseOrUnit_Error', 'message': '企业/单位名称过长'})
        if len(UserPosition) >= 100:
            errors.append({'field': 'UserPosition_Error', 'message': '职位名称过长'})
        if len(UnitKind) >= 100:
            errors.append({'field': 'UnitKind_Error', 'message': '企业性质名称过长'})

        if errors:  # 拒绝访问数据库
            return jsonify({'status': 'error', 'errors': errors}), 400
        else:
            stored_code = check_expiration(Email)
            if stored_code is None:
                errors.append({'field': 'code', 'message': 'code haven`t sent'})
                return jsonify({'status': 'error', 'errors': errors}), 400
            else:
                if code == stored_code:
                    result = register_db(Email, EnterpriseOrUnit, UserPosition, UnitKind)
                    if result['status'] == 'error':
                        errors.append({'field': 'Email_ExistedError', 'message': result['message']})  # 邮箱已被注册
                        return jsonify({'status': 'error', 'errors': errors}), 400
                    else:
                        delete_code(Email)  # 清除验证码数据库缓存
                        # 设置会话数据
                        session['Email'] = Email  # 唯一键
                        return jsonify({'status': 'success', 'errors': errors}), 200
                else:
                    errors.append({'field': 'code', 'message': 'the verification code is incorrect'})
                    return jsonify({'status': 'error', 'errors': errors}), 400


@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.json
    if data is None:
        return jsonify({'status': 'error', 'message': 'Request_Error'}), 400
    else:
        email = data['email']
        code = data['code']
        flag = 1  # 1 成功登录， 0 登录失败
        email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
        if not re.match(email_regex, email):
            flag = 0
        if flag == 0:
            return jsonify({'status': 'error', 'message': 'illegal email address'}), 400
        else:
            user = get_user(email)
            if user:
                stored_code = check_expiration(email)
                if stored_code is None:
                    return jsonify({'status': 'error', 'message': 'code haven`t sent'}), 400
                else:
                    if code == stored_code:
                        session['Email'] = email
                        delete_code(email)  # 清除验证码数据库缓存
                        return jsonify({'status': 'success', 'message': ''}), 200
                    else:
                        return jsonify({'status': 'error', 'message': 'the verification code is incorrect'}), 400
            else:
                return jsonify({'status': 'email_error', 'message': 'user does not exist'}), 400


@auth_bp.route('/test_login', methods=['POST'])
def test_login():
    data = request.json
    if data is None:
        return jsonify({'status': 'error', 'message': 'Request_Error'}), 400
    else:
        email = data['email']
        session['Email'] = email
        logger.info(f"用户{session['Email']}登录成功")
        return jsonify({'status': 'success', 'message': 'login'}), 200


@auth_bp.route('/logout', methods=['GET'])
def logout():
    logger.info(f"用户{session['Email']}已退出")
    session.clear()  # 清除**服务端**会话数据，不会清除客户端cookie

    # 设置一个响应，并让 session cookie 立即过期
    response = make_response(jsonify({'status': 'success'}))
    response.set_cookie('session', '', expires=0)  # 将 session cookie 删除
    logger.info(session)
    return response


@auth_bp.route('/check_login', methods=['GET'])  # 检查用户是否已登录
def check_login():
    if 'Email' in session:
        return jsonify(logged_in=True)
    else:
        return jsonify(logged_in=False)