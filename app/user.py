from flask import session
from .models import get_user
from .config import Config  # 导入 Config 类
import sqlite3

class User:
    def __init__(self, email):
        self.email = email
        self.user_dict = get_user(self.email)

    def get_userId(self):
        return self.user_dict['Id']

    def count_task_all(self, UserId):
        """
        :param UserId: 用户的ID
        :return: 该用户所有类型的记录总数, 出错时返回 -1
        """
        conn = sqlite3.connect(Config.DATABASE_URI)
        c = conn.cursor()
        try:
            c.execute('''
                SELECT COUNT(*) FROM refractivityFile 
                WHERE UserId = ?
            ''', (UserId,))
            count = c.fetchone()[0]
            return count
        except sqlite3.Error as e:
            return -1  # 出错时返回 -1
        finally:
            conn.close()

    def count_task(self, UserId, type_id):
        """
               :param UserId: 用户的ID
               :param type_id: 光学0 电学1... (对应TypeId字段)
               :return: 该用户该type的记录数  出错时返回 -1
               """
        conn = sqlite3.connect(Config.DATABASE_URI)
        c = conn.cursor()
        try:
            c.execute('''
                       SELECT COUNT(*) FROM refractivityFile 
                       WHERE TypeId = ? AND UserId = ?
                   ''', (type_id, UserId))
            count = c.fetchone()[0]
            return count
        except sqlite3.Error as e:
            return -1
        finally:
            conn.close()

    def delete_task(self, UserId, type_id):
        """
                :param UserId: 用户的ID
                :param type_id: 光学0 电学1... (对应TypeId字段)
                :return: True：成功删除 False：删除失败
                """
        conn = sqlite3.connect(Config.DATABASE_URI)
        c = conn.cursor()
        try:
            c.execute('''
                        DELETE FROM refractivityFile 
                        WHERE Id = (SELECT Id FROM refractivityFile 
                                    WHERE TypeId = ? AND UserId = ? 
                                    ORDER BY Id ASC LIMIT 1)
                    ''', (type_id, UserId))
            conn.commit()
            return True
        except sqlite3.Error as e:
            return False
        finally:
            conn.close()

    def store_task_file(self, UserId, type_id, path):
        """
                :param UserId: 用户的ID
                :param type_id: 光学0 电学1... (对应TypeId字段)
                :param path: 任务实例的存储文件路径(字符串) (对应FilePath字段)
                :return: True：成功存储 False：存储失败
                """
        # 先检查是否已有十条记录
        if self.count_task(UserId, type_id) >= 10:
            self.delete_task(UserId, type_id)

        # 插入新的记录
        conn = sqlite3.connect(Config.DATABASE_URI)
        c = conn.cursor()
        try:
            c.execute('''
                       INSERT INTO refractivityFile (TypeId, UserId, FilePath) 
                       VALUES (?, ?, ?)
                   ''', (type_id, UserId, path))
            conn.commit()
            return True
        except sqlite3.Error as e:
            return False
        finally:
            conn.close()