from .config import Config  # 导入 Config 类
import sqlite3
from datetime import datetime, timedelta

from flask import Flask
from .config import Config  # 导入 Config 类
import sqlite3
import hashlib


def init_db():
    conn = sqlite3.connect(Config.DATABASE_URI)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            Id INTEGER PRIMARY KEY AUTOINCREMENT,
            Email TEXT NOT NULL UNIQUE,
            EnterpriseOrUnit TEXT NULL,
            UserPosition TEXT NULL,
            UnitKind TEXT NULL
        );
    ''')
    conn.commit()
    c.execute('''
        CREATE TABLE IF NOT EXISTS verifyCode (
            Id INTEGER PRIMARY KEY AUTOINCREMENT,
            Email TEXT NOT NULL,
            code TEXT NOT NULL,
            expiration DATETIME NOT NULL
        )
    ''')
    conn.commit()
    conn.close()


def register_db(Email, EnterpriseOrUnit, UserPosition, UnitKind):
    conn = sqlite3.connect(Config.DATABASE_URI)
    c = conn.cursor()
    try:
        c.execute('''
            INSERT INTO users (Email, EnterpriseOrUnit, UserPosition, UnitKind)
            VALUES (?, ?, ?, ?)
        ''', (Email, EnterpriseOrUnit, UserPosition, UnitKind))
        conn.commit()
        return {"status": "success", "message": "用户成功注册"}
    except sqlite3.IntegrityError:
        return {"status": "error", "message": "email已存在"}
    finally:
        conn.close()

def get_user(email):
    conn = sqlite3.connect(Config.DATABASE_URI)
    c = conn.cursor()
    try:
        c.execute('''
            SELECT * FROM users WHERE email = ?
        ''', (email,))
        user = c.fetchone()
        if user:
            user_dict = {
                "Id": user[0],
                "Email": user[1],
                "EnterpriseOrUnit": user[2],
                "UserPosition": user[3],
                "UnitKind": user[4]
            }
            return user_dict
        else:
            return None
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None
    finally:
        conn.close()

# 存储验证码和过期时间
def store_verification_code(email, code):
    expiration = datetime.utcnow() + timedelta(minutes=Config.CODE_DURATION_TIME)  # 当前UTC时间加有效时长
    conn = sqlite3.connect(Config.DATABASE_URI)
    c = conn.cursor()
    c.execute('''
        INSERT OR REPLACE INTO verifyCode (email, code, expiration)
        VALUES (?, ?, ?)
    ''', (email, code, expiration))
    conn.commit()
    conn.close()
# 成功登录后删除用户下的所有验证码
def delete_verification_code(email):
    conn = sqlite3.connect(Config.DATABASE_URI)
    c = conn.cursor()
    c.execute('''
        DELETE FROM verifyCode
        WHERE Email = ?
    ''', (email,))
    conn.commit()
    conn.close()
# 核验是否有已发送并且未过期的验证码
def check_expiration(email):
    conn = sqlite3.connect(Config.DATABASE_URI)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT code, expiration 
        FROM verifyCode 
        WHERE email = ? AND expiration >= CURRENT_TIMESTAMP
    ''', (email,))
    result = cursor.fetchone()
    conn.close()
    if result is None:
        return None
    else:
        return result[0]  # 返回查询到的验证码内容
def delete_code(email):
    conn = sqlite3.connect(Config.DATABASE_URI)
    cursor = conn.cursor()

    # 执行删除操作
    cursor.execute('''
            DELETE FROM verifyCode 
            WHERE email = ?
        ''', (email,))

    # 提交事务
    conn.commit()
    # 关闭数据库连接
    conn.close()
