import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd
import logging
import os
import json
import torch.nn.functional as F
from tqdm import tqdm

# 设置可见的CUDA设备
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("../log/proxy_google_evaluate.log", mode='w')
    ]
)

# 选择设备
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device: {device}')

# 定义DNN模型
class DNNClassifier(nn.Module):
    def __init__(self):
        super(DNNClassifier, self).__init__()
        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)  # 16个新的moderation标签
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.fc5(x)
        return x

# 创建数据集类
class PromptDataset(torch.utils.data.Dataset):
    def __init__(self, prompts, embeddings, labels, categories):
        self.prompts = prompts
        self.embeddings = embeddings
        self.labels = labels
        self.categories = categories

    def __len__(self):
        return len(self.categories)

    def __getitem__(self, idx):
        return self.prompts[idx], self.embeddings[idx], self.labels[idx], self.categories[idx]


def generate_json_result(prompt, outputs, predicted_categories):
    predicted_categories = predicted_categories.tolist()  # 将张量转换为标量列表
    
    categories_dict = {
        "Toxic": bool(predicted_categories[0]),
        "Insult": bool(predicted_categories[1]),
        "Profanity": bool(predicted_categories[2]),
        "Derogatory": bool(predicted_categories[3]),
        "Sexual": bool(predicted_categories[4]),
        "Death, Harm & Tragedy": bool(predicted_categories[5]),
        "Violent": bool(predicted_categories[6]),
        "Firearms & Weapons": bool(predicted_categories[7]),
        "Public Safety": bool(predicted_categories[8]),
        "Health": bool(predicted_categories[9]),
        "Religion & Belief": bool(predicted_categories[10]),
        "Illicit Drugs": bool(predicted_categories[11]),
        "War & Conflict": bool(predicted_categories[12]),
        "Politics": bool(predicted_categories[13]),
        "Finance": bool(predicted_categories[14]),
        "Legal": bool(predicted_categories[15])
    }
    
    category_scores_dict = {
        "Toxic": format(outputs[0][0].item(), ".15f"),
        "Insult": format(outputs[0][1].item(), ".15f"),
        "Profanity": format(outputs[0][2].item(), ".15f"),
        "Derogatory": format(outputs[0][3].item(), ".15f"),
        "Sexual": format(outputs[0][4].item(), ".15f"),
        "Death, Harm & Tragedy": format(outputs[0][5].item(), ".15f"),
        "Violent": format(outputs[0][6].item(), ".15f"),
        "Firearms & Weapons": format(outputs[0][7].item(), ".15f"),
        "Public Safety": format(outputs[0][8].item(), ".15f"),
        "Health": format(outputs[0][9].item(), ".15f"),
        "Religion & Belief": format(outputs[0][10].item(), ".15f"),
        "Illicit Drugs": format(outputs[0][11].item(),".15f"),
        "War & Conflict": format(outputs[0][12].item(),".15f"),
        "Politics": format(outputs[0][13].item(),".15f"),
        "Finance": format(outputs[0][14].item(),".15f"),
        "Legal": format(outputs[0][15].item(),".15f")
    }


    json_result = {
        "flagged": any(predicted_categories),  # 检查是否有任意类别被标记
        "categories": categories_dict,
        "category_scores": category_scores_dict,
        "prompt": prompt
    }

    return json_result

# 加载JSONL数据
def load_jsonl_data(jsonl_file):
    category_names = [
        "Toxic", "Insult", "Profanity", "Derogatory", "Sexual",
        "Death, Harm & Tragedy", "Violent", "Firearms & Weapons",
        "Public Safety", "Health", "Religion & Belief", "Illicit Drugs",
        "War & Conflict", "Politics", "Finance", "Legal"
    ]
    categories = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            category_dict = {item['name']: item['confidence'] for item in data['moderationCategories']}
            category_values = [category_dict.get(name, 0.0) for name in category_names]
            categories.append(torch.tensor(category_values, dtype=torch.float))
    return categories

def train_and_save_model():

    file = "advbench_role_100_AutoDAN_Liu"

    if file == "advbench_100" or file == "advbench_520" or file in ["DNA_100","DNA_939","harmBench_100","harmBench_320"]:
        colmn = "harmful"
        embedding_file = f'/data/home/Xinzhe/GuardBreaker/guard_attack/data/embedding/Llama_3_8b/-1/{file}_{colmn}.pt'  # 正确的字符串路径
        csv_file = f'/data/home/Xinzhe/GuardBreaker/guard_attack/data/csv/{file}.csv'   
        jsonl_file = f"/data/home/Xinzhe/GuardBreaker/guard_attack/data/score/{file}_google.jsonl" 

    elif "role" in file:
        colmn = "prompt"
        # embedding_file = f'/data/home/Xinzhe/GuardBreaker/guard_attack/data/embedding/Llama_3_8b/-1/{file}_{colmn}.pt'  # 正确的字符串路径
        embedding_file = f'/data/home/Xinzhe/GuardBreaker/baseline/AutoDAN_Liu/embedding/Llama_3_8b/-1/{file}_{colmn}.pt'  # 正确的字符串路径
        # csv_file = f'/data/home/Xinzhe/GuardBreaker/guard_attack/data/csv/{file}.csv'
        csv_file = f'/data/home/Xinzhe/GuardBreaker/baseline/AutoDAN_Liu/data/{file}.csv'
        # jsonl_file = f"/data/home/Xinzhe/GuardBreaker/guard_attack/data/score/{file}_google.jsonl" 
        jsonl_file = f"/data/home/Xinzhe/GuardBreaker/baseline/AutoDAN_Liu/score/{file}_google.jsonl" 

    elif 'best' in file:    
        colmn = "prompt_with_adv"
        embedding_file = f'/data/home/Xinzhe/GuardBreaker/guard_attack/data/embedding/Llama_3_8b/-1/{file}_{colmn}.pt'  # 正确的字符串路径
        csv_file = f'/data/home/Xinzhe/GuardBreaker/guard_attack/data/csv/{file}.csv'
        jsonl_file = f"/data/home/Xinzhe/GuardBreaker/guard_attack/data/score/{file}_google.jsonl" 
    elif file == "harmful_contents":
        colmn = "prompts"
        embedding_file = f'/data/home/Xinzhe/GuardBreaker/proxy_model/data/embedding/Llama-3-8b/{file}_{colmn}.pt'  # 正确的字符串路径
        csv_file = f'/data/home/Xinzhe/GuardBreaker/proxy_model/data/csv/{file}.csv'      
        jsonl_file = f"/data/home/Xinzhe/GuardBreaker/guard_attack/data/score/{file}_google.jsonl"
    elif "prp" in file:
        colmn = "prompt_with_adv"
        embedding_file = f'/data/home/Xinzhe/GuardBreaker/baseline/PRP/embedding/Llama_3_8b/-1/{file}_{colmn}.pt'
        csv_file = f'/data/home/Xinzhe/GuardBreaker/baseline/PRP/data/{file}.csv'
        jsonl_file = f'/data/home/Xinzhe/GuardBreaker/baseline/PRP/score/{file}_google.jsonl'

    output_file = f'/data/home/Xinzhe/GuardBreaker/baseline/PRP/score/{file}_proxy_google.jsonl'


    embedding_data = torch.load(embedding_file)  # 从字符串路径加载数据
    prompts = pd.read_csv(csv_file)[colmn].tolist()
    labels = torch.ones(len(prompts), dtype=torch.long)
    categories = load_jsonl_data(jsonl_file)

    # 创建数据集
    dataset = PromptDataset(prompts, embedding_data, labels, categories)

    test_loader = DataLoader(dataset, batch_size=1)

    model = DNNClassifier().to(device)
    model.load_state_dict(torch.load("../model/proxy_google.pth"))
    logging.info("ini over")
    model.eval()

    TT_cnt = 0
    TF_cnt = 0
    FT_cnt = 0
    FF_cnt = 0
    total = 0
    
    with open(output_file, 'a') as jsonl_file:  # 使用文件句柄而不是字符串
        with torch.no_grad():
            for i, (prompts, embeddings, label, categories) in tqdm(enumerate(test_loader), desc="Evaluating"):
                
                total += 1

                embeddings = embeddings.to(device)
                label = label.to(device)
                categories = categories.to(device)
                
                outputs = model(embeddings)
                outputs = F.sigmoid(outputs)
                predicted_categories = (outputs > 0.5).float()
                categories = (categories > 0.5).float()


                # 计算当前的flag状态
                predicted_flagged = predicted_categories.any().item()
                
                # 计算预期的flag状态
                target_flagged = categories.any().item()     

                # 生成结果
                json_result = generate_json_result(prompts[0], outputs, predicted_categories.squeeze())  # 修复此处为 prompts[0]

                # 写入到jsonl文件中
                json.dump(json_result, jsonl_file)
                jsonl_file.write('\n')
    
                if int(predicted_flagged) == label.item(): 
                    if predicted_flagged == target_flagged:
                        TT_cnt += 1
                    else:
                        TF_cnt += 1
                else:
                    if predicted_flagged == target_flagged:
                        FT_cnt += 1
                    else:
                        FF_cnt += 1

    logging.info(f"代理模型认为有害而且google API预测为有害: {TT_cnt}")
    logging.info(f"代理模型认为有害但是google API预测为无害: {TF_cnt}")
    logging.info(f"代理模型认为无害而且google API预测为无害: {FT_cnt}")
    logging.info(f"代理模型认为无害但是google API预测为有害: {FF_cnt}")

    logging.info(f"代理模型认为有害的数量: {TT_cnt + TF_cnt}")
    logging.info(f"代理模型认为有害的比例: {(TT_cnt + TF_cnt) / total}")
    logging.info(f"google API预测为有害的数量: {TT_cnt + FF_cnt}")
    logging.info(f"google API预测为无害的数量: {TF_cnt + FT_cnt}")


if __name__ == "__main__":
    train_and_save_model()    