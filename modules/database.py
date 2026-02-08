import sqlite3
import pandas as pd
import datetime
import json
import os

class PatientDatabase:
    def __init__(self, db_name="patient_records.db"):
        self.db_name = db_name
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        # 通用表结构：inputs 存为 JSON 字符串以适应任何特征组合
        c.execute('''
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                inputs TEXT,
                risk_prob REAL,
                risk_label TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def add_record(self, inputs_dict, prob, label):
        try:
            conn = sqlite3.connect(self.db_name)
            c = conn.cursor()
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 将字典转为 JSON 字符串
            inputs_str = json.dumps(inputs_dict)
            
            c.execute("INSERT INTO records (timestamp, inputs, risk_prob, risk_label) VALUES (?, ?, ?, ?)",
                      (timestamp, inputs_str, prob, label))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"DB Error: {e}")

    def fetch_all_records(self):
        try:
            conn = sqlite3.connect(self.db_name)
            df = pd.read_sql_query("SELECT * FROM records ORDER BY timestamp DESC", conn)
            conn.close()
            
            if not df.empty and 'inputs' in df.columns:
                # 尝试解析 JSON 并展开为列
                try:
                    inputs_df = pd.json_normalize(df['inputs'].apply(json.loads))
                    df = pd.concat([df.drop('inputs', axis=1), inputs_df], axis=1)
                except:
                    pass # 如果解析失败，保留原样
            return df
        except Exception:
            return pd.DataFrame()
