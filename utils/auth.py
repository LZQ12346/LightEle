from app.config import Config  # 导入 Config 类
from email.mime.text import MIMEText
from email.utils import formataddr
import smtplib
import random
import string


def send_verification_email(to_addr, code):
    # SMTP服务器以及相关配置信息
    smtp_sever = Config.SMTP_SERVER
    from_addr = Config.FROM_ADDR
    password = Config.PASSWORD # 授权码
    try:
        #  1.创建邮件(写好邮件内容、发送人、收件人和标题等)
        msg = MIMEText(
            f"""
            <html>
                <head></head>
                <body>
                    <table width="100%" cellpadding="0" cellspacing="0" border="0" style="background-color: #f9f9f9; padding: 20px;">
                        <tr>
                            <td align="center">
                                <table width="600" cellpadding="0" cellspacing="0" border="0" style="background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);">
                                    <tr>
                                        <td style="font-size: 24px; font-weight: bold; margin-bottom: 20px; text-align: left;">尊敬的用户，</td>
                                    </tr>
                                    <tr>
                                        <td style="font-size: 16px; text-align: left; padding-bottom: 20px;">
                                            <p>感谢您注册 <strong>苏州大学仿真计算平台</strong>。</p>
                                            <p>您的验证码是：</p>
                                            <p style="font-size: 24px; font-weight: bold; color: #ff4c4c;">{code}</p>
                                            <p>验证码两分钟内有效。</p>
                                            <p>请在验证表单中输入此验证码以完成注册和登录，若非本人操作，请忽略此邮件。</p>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td style="font-size: 14px; color: #777; text-align: left; padding-top: 20px; border-top: 1px solid #eeeeee;">
                                            <p>苏州大学仿真计算开发团队</p>
                                            <p>联系邮箱：<a href="19326488034@163.com" style="color: #333333;">19326488034@163.com</a></p>
                                        </td>
                                    </tr>
                                </table>
                            </td>
                        </tr>
                    </table>
                </body>
            </html>
            """,
            'html',
            'utf-8'
        )
        msg['From'] = formataddr(('苏州大学仿真计算', from_addr))  # 发件人昵称和邮箱
        msg['To'] = to_addr # 收件人昵称和邮箱
        msg['Subject'] = f"您的验证码：{code}" # 邮件标题
        # 2.登录账号
        server = smtplib.SMTP_SSL(smtp_sever, 465) # 加密通信
        server.login(from_addr, password)
        # 3.发送邮件
        server.sendmail(from_addr, to_addr, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False


def generate_verification_code(length=4):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))
