import pandas as pd
import torch
import xgboost as xgb
import jsonlines
import json
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("./log/xgboost_testing.log", mode='w')
    ]
)

file_prefix = "/data/home/Xinzhe/GuardBreaker_new/data/score/guard_attack"
file_name = "harmBench_scenario_gcg_guard3_100"

# 加载测试数据中的护栏分数
def load_advbench_scores():
    scores = {
        "guard3": [],
        "guardrails": [],
        "nemo": [],
        "openai": [],
        "google": []
    }

    def read_jsonl_with_error_handling(filepath, key, condition_func):
        with open(filepath, "r", encoding="utf8") as f:
            for i, line in enumerate(f):
                try:
                    # 逐行检查并解析 JSON 数据
                    parsed_line = json.loads(line.strip())
                    if condition_func(parsed_line):
                        scores[key].append(1)
                    else:
                        scores[key].append(0)
                except json.JSONDecodeError as e:
                    logging.warning(f"Invalid JSON at line {i+1} in {filepath}: {line.strip()} - Error: {e}")
                except KeyError as e:
                    logging.warning(f"Missing key in JSON at line {i+1} in {filepath}: {line.strip()} - Error: {e}")

    # 读取各护栏分数文件
    read_jsonl_with_error_handling(
        f"{file_prefix}/{file_name}_guard3.jsonl",
        "guard3",
        lambda item: item.get("result") == "unsafe"
    )
    read_jsonl_with_error_handling(
        f"{file_prefix}/{file_name}_guardrails.jsonl",
        "guardrails",
        lambda item: item.get("result") == 1
    )
    read_jsonl_with_error_handling(
        f"{file_prefix}/{file_name}_nemo.jsonl",
        "nemo",
        lambda item: item.get("result") == 1
    )
    read_jsonl_with_error_handling(
        f"{file_prefix}/{file_name}_openai.jsonl",
        "openai",
        lambda item: item.get("flagged", False)
    )
    read_jsonl_with_error_handling(
        f"{file_prefix}/{file_name}_google.jsonl",
        "google",
        lambda item: any(cc.get("confidence", 0) > 0.5 for cc in item.get("moderationCategories", []))
    )
    
    return scores

def prepare_test_data():
    # 读取分数数据
    scores = load_advbench_scores()
    
    # 确保所有评分列表长度一致，避免因缺失数据导致后续步骤出错
    min_length = min(len(scores[key]) for key in scores)
    logging.info(f"Minimum length of scores: {min_length}")
    
    # 修正长度不一致问题
    for key in scores:
        scores[key] = scores[key][:min_length]
    
    # 将每条样本的分数组合成模型输入的格式
    data = []
    for i in range(min_length):
        sample = [
            scores["guard3"][i],
            scores["guardrails"][i],
            scores["nemo"][i],
            scores["openai"][i],
            scores["google"][i]
        ]
        data.append(sample)
    
    return np.array(data)

# 加载训练好的模型
model = xgb.XGBClassifier()
model.load_model('./model/xgboost_model.json')

# 准备测试数据
X_test = prepare_test_data()

# 进行预测
y_pred = model.predict(X_test)

# 统计并输出预测为 harmful 和 normal 的数量
harmful_count = np.sum(y_pred == 1)
normal_count = np.sum(y_pred == 0)

# 逐条输出每个样本的预测结果
for i, prediction in enumerate(y_pred):
    logging.info(f"Sample {i+1}: Prediction - {'Harmful' if prediction == 1 else 'Not Harmful'}")

logging.info(f"{file_prefix}/{file_name}.csv:")
logging.info(f"Predictions for {file_name}.csv:")
logging.info(f"Harmful predictions: {harmful_count}")
logging.info(f"Normal predictions: {normal_count}")
logging.info(f"Normal : {normal_count} / {len(y_pred)} = {normal_count / len(y_pred)}")
