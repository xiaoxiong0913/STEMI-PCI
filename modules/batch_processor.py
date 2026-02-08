import pandas as pd
import numpy as np
import io

class BatchProcessor:
    def __init__(self, model, scaler, imputer=None):
        self.model = model
        self.scaler = scaler
        self.imputer = imputer
        # STEMI 模型的 8 个标准输入特征 (必须与训练时一致)
        self.expected_cols = [
            'Age', 'Hb', 'AST', 
            'Respiratory support', 'Beta blocker', 'Cardiotonics', 
            'Statins', 'Stent for IRA'
        ]
        # 定义需要标准化的连续变量 (与 scaler.pkl 训练时的列一致)
        self.cont_cols = ['Age', 'Hb', 'AST']
        # 定义不需要标准化的分类变量
        self.cat_cols = ['Respiratory support', 'Beta blocker', 'Cardiotonics', 'Statins', 'Stent for IRA']

    def process_data(self, df):
        # 1. 检查列名完整性
        missing = [c for c in self.expected_cols if c not in df.columns]
        if missing:
            return None, f"Missing columns in uploaded file: {missing}"
        
        try:
            # 2. 提取数据
            # 确保分类变量是数值型 (0/1/2)
            for col in self.cat_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            
            # 3. 预处理 (混合逻辑)
            # Part A: 连续变量 -> 标准化
            X_cont = df[self.cont_cols].values
            if self.imputer:
                # 如果有缺失值处理器，先处理连续变量的缺失值
                # 注意：这里假设 imputer 是针对连续变量训练的，如果不是，需调整逻辑
                # 鉴于您的 scaler.pkl 只包含 3 个变量，我们假设输入数据完整或已处理
                pass 
            
            X_cont_scaled = self.scaler.transform(X_cont)
            
            # Part B: 分类变量 -> 保持原始数值
            X_cat = df[self.cat_cols].values
            
            # Part C: 拼接 (水平堆叠)
            # 最终形状: (n_samples, 3 + 5) = (n_samples, 8)
            X_final = np.hstack((X_cont_scaled, X_cat))
            
            # 4. 预测 (GBM predict_proba)
            # 获取正类 (Class 1: 死亡) 的概率
            probs = self.model.predict_proba(X_final)[:, 1]
            
            # 5. 构造结果
            res_df = df.copy()
            res_df['Predicted_3Yr_Mortality'] = probs
            # 阈值 0.147 来自手稿 Youden Index
            res_df['Risk_Group'] = res_df['Predicted_3Yr_Mortality'].apply(lambda x: "High Risk" if x >= 0.147 else "Low Risk")
            
            return res_df, None
            
        except Exception as e:
            return None, f"Processing Error: {str(e)}"

    def convert_to_excel(self, df):
        """辅助函数：将结果转为 Excel 字节流供下载"""
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Predictions')
        return output.getvalue()
