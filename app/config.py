import os

class Config:
    # flask
    SECRET_KEY = 'ylab_LightEle'  # 用于 Flask 的安全密钥,用于加密session数据
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))

    # 注意None必须和https绑定， Lax可以在http使用cookie，Strict则不允许使用cookie
    SESSION_COOKIE_SAMESITE = 'None'  # Lax None Strict
    SESSION_COOKIE_SECURE = True  # 是否使用https，本地测试使用HTTP时，设置为False

    SESSION_PERMANENT = True
    # database
    DATABASE_URI = os.path.join(BASE_DIR, "..", 'instance', 'LightEle.db')

    # SMTP服务器以及相关配置信息
    SMTP_SERVER = 'smtp.163.com'
    FROM_ADDR = "19326488034@163.com"
    PASSWORD = "ZRDGTFQWAXNZHGIL"  # 授权码
    CODE_DURATION_TIME = 3 # 验证码有效期，单位：分钟
