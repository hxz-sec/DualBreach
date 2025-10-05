import torch
import random
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging
import jsonlines

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("./log/xgboost_training_new.log", mode='w')
    ]
)

file_path = "/data/home/Xinzhe/GuardBreaker_new/github/EGuard/data/"


# 定义数据加载函数
def LlamaGuard3_data():
    hp = torch.load(f'{file_path}/harmful_contents_2k_guard3.pt')
    np = torch.load(f'{file_path}/normal_contents_2k_guard3.pt')
    return hp, np

def OpenAI_data(N=2000):
    hp = torch.zeros(N, 2)
    np = torch.zeros(N, 2)
    with open(f"{file_path}/openai-normal-2k.jsonl", "r+", encoding="utf8") as f:
        i = 0
        for item in jsonlines.Reader(f):
            if i >= N:
                break
            if item["flagged"]:
                np[i][1] = 1
            if not item["flagged"]:
                np[i][0] = 1
            i += 1
    with open(f"{file_path}/openai-harmful-2k.jsonl", "r+", encoding="utf8") as f:
        i = 0
        for item in jsonlines.Reader(f):
            if i >= N:
                break
            if item["flagged"]:
                hp[i][1] = 1
            if not item["flagged"]:
                hp[i][0] = 1
            i += 1
    return hp, np

def Guardrails_data():
    hp = torch.load(f"{file_path}/Llama-2-7b-hf/-1/gr-harmful-2k.pt")
    np = torch.load(f"{file_path}/Llama-2-7b-hf/-1/gr-normal-2k.pt")
    return hp, np

def Google_data(N=2000):
    hp = torch.zeros(N, 2)
    np = torch.zeros(N, 2)
    with open(f"{file_path}/normal_google.jsonl", "r+", encoding="utf8") as f:
        i = 0
        for item in jsonlines.Reader(f):
            c = item["moderationCategories"]
            for cc in c:
                if cc["confidence"] > 0.5:
                    np[i][1] = 1
                    break
            if not np[i][1]:
                np[i][0] = 1
            i += 1
    with open(f"{file_path}/harmful_google.jsonl", "r+", encoding="utf8") as f:
        i = 0
        for item in jsonlines.Reader(f):
            c = item["moderationCategories"]
            for cc in c:
                if cc["confidence"] > 0.5:
                    hp[i][1] = 1
                    break
            if not hp[i][1]:
                hp[i][0] = 1
            i += 1
    return hp, np

def Nemo_data(N=2000):
    hp = torch.zeros(N, 2)
    np = torch.zeros(N, 2)
    with open(f"{file_path}/normal_nemo.jsonl", "r+", encoding="utf8") as f:
        i = 0
        for item in jsonlines.Reader(f):
            if item["result"]:
                np[i][1] = 1
            if not item["result"]:
                np[i][0] = 1
            i += 1
    with open(f"{file_path}/harmful_nemo.jsonl", "r+", encoding="utf8") as f:
        i = 0
        for item in jsonlines.Reader(f):
            if item["result"]:
                hp[i][1] = 1
            if not item["result"]:
                hp[i][0] = 1
            i += 1
    return hp, np

# 动态调整权重函数
# 动态调整权重函数
def adjust_weights(sample):
    guard_1 = sample[0][1]
    weights = [0, 0.25, 0.25, 0.25, 0.25]  # 默认分配权重给其他四个 Guard

    if guard_1 >= 0.5:
        # Guard 1 识别成功，使用 Guard 1 的权重（全部归于 Guard 1）
        weights = [1, 0, 0, 0, 0]
    else:
        # Guard 1 识别失败，平分权重给其他四个 Guard
        weights = [0, 0.25, 0.25, 0.25, 0.25]

    # 根据权重调整置信度
    return [sample[i][1] * weights[i] for i in range(5)]


# 加载数据并动态调整权重
def load_confidence_data():
    harmful_data_p_1, normal_data_p_1 = LlamaGuard3_data()
    harmful_data_p_2, normal_data_p_2 = Google_data()
    harmful_data_p_3, normal_data_p_3 = Guardrails_data()
    harmful_data_p_4, normal_data_p_4 = Nemo_data()
    harmful_data_p_5, normal_data_p_5 = OpenAI_data()

    harmful_data, normal_data = [], []
    for harmful_sample, normal_sample in zip(
        zip(harmful_data_p_1, harmful_data_p_2, harmful_data_p_3, harmful_data_p_4, harmful_data_p_5),
        zip(normal_data_p_1, normal_data_p_2, normal_data_p_3, normal_data_p_4, normal_data_p_5)
    ):
        # 判定权重： Guard 1 识别成功则偏向 Guard 3 权重，否则用其他 Guard 权重
        harmful_weights = adjust_weights(harmful_sample)
        normal_weights = adjust_weights(normal_sample)
        harmful_data.append(harmful_weights)
        normal_data.append(normal_weights)

    harmful_data = torch.tensor(harmful_data)
    normal_data = torch.tensor(normal_data)
    harmful_labels = torch.ones(harmful_data.size(0))
    normal_labels = torch.zeros(normal_data.size(0))

    return torch.cat((harmful_data, normal_data), dim=0), torch.cat((harmful_labels, normal_labels), dim=0)

# 数据准备
X, y = load_confidence_data()
X = X.numpy()
y = y.numpy()

# 将数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用XGBoost模型进行训练
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# logloss--负对数似然函数（二分类）



# 保存训练好的XGBoost模型
model.save_model('./model/xgboost_model_new.json')
logging.info("Model saved as xgboost_model_new.json")


# 测试模型
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
logging.info(f"Accuracy on test set: {accuracy:.2f}%")

# 输出每个护栏的权重（特征重要性）
feature_importance = model.feature_importances_
logging.info("Feature Importance for each guard:")
for i, importance in enumerate(feature_importance):
    logging.info(f"Guard {i+1} importance: {importance:.4f}")



