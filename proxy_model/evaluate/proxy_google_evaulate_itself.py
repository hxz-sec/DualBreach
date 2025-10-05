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

# 设置可见的CUDA设备
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("../log/training_proxy_google_evaluate.log", mode='w')
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
        self.dropout = nn.Dropout(p=0.5)
        
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
    def __init__(self, prompts, embeddings, categories):
        self.prompts = prompts
        self.embeddings = embeddings
        self.categories = categories

    def __len__(self):
        return len(self.categories)

    def __getitem__(self, idx):
        return self.prompts[idx], self.embeddings[idx], self.categories[idx]

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

    # 实际训练
    # data_limits = {
    #     "harmful": 800,
    #     "trivia_qa_train": 200,
    #     "openbookqa": 200,
    #     "filtered_wiki_qa": 200,
    #     "yelp_train": 200
    # }

    # 定义每个数据集要加载的数据数量（测试用）
    data_limits = {
        "harmful": 70000,
        "trivia_qa_train": 32000,
        "openbookqa": 5000,
        "filtered_wiki_qa": 1000,
        "yelp_train": 32000
    }

    # 定义数据集路径
    data_paths = {
        "harmful": {
            "embedding": '../data/embedding/Llama-3-8b/harmful_contents_prompts.pt',
            "csv": '../data/csv/harmful_contents.csv',
            "jsonl": '../data/score/harmful_contents_75000_google.jsonl'
        },
        "trivia_qa_train": {
            "embedding": '../data/embedding/Llama-3-8b/trivia_qa_train_prompts.pt',
            "csv": '../data/csv/trivia_qa_train.csv',
            "jsonl": '../data/score/trivia_qa_train_35000_google.jsonl'
        },
        "openbookqa": {
            "embedding": '../data/embedding/Llama-3-8b/openbookqa_prompts.pt',
            "csv": '../data/csv/openbookqa.csv',
            "jsonl": '../data/score/openbookqa_5000_google.jsonl'
        },
        "filtered_wiki_qa": {
            "embedding": '../data/embedding/Llama-3-8b/filtered_wiki_qa_prompts.pt',
            "csv": '../data/csv/filtered_wiki_qa.csv',
            "jsonl": '../data/score/filtered_wiki_qa_1000_google.jsonl'
        },
        "yelp_train": {
            "embedding": '../data/embedding/Llama-3-8b/yelp_train_prompts.pt',
            "csv": '../data/csv/yelp_train.csv',
            "jsonl": '../data/score/yelp_train_35000_google.jsonl'
        }
    }

    # 加载和合并数据
    all_prompts = []
    all_embeddings = []
    all_categories = []

    for key, paths in data_paths.items():
        # 加载嵌入
        embedding_data = torch.load(paths['embedding'])[:data_limits[key]]
        # 加载提示文本
        prompts = pd.read_csv(paths['csv'])["prompts"].tolist()[:data_limits[key]]
        # 加载类别
        categories = load_jsonl_data(paths['jsonl'])[:data_limits[key]]

        all_prompts.extend(prompts)
        all_embeddings.append(embedding_data)
        all_categories.extend(categories)

    # 合并所有的嵌入和类别
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_categories = torch.stack(all_categories)

    # 创建数据集
    dataset = PromptDataset(all_prompts, all_embeddings, all_categories)

    # 划分数据集为训练集和测试集
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 打印训练集和测试集的大小
    logging.info(f"Total prompts count: {len(all_prompts)}")
    logging.info(f"Training dataset size: {train_size}")
    logging.info(f"Testing dataset size: {test_size}")


    batch_size = 4
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = DNNClassifier().to(device)
    model.load_state_dict(torch.load("../model/proxy_google.pth"))
    print("ini over")

    # 使用测试集评估模型
    model.eval()
    total_correct = 0
    total_samples = 0
    harmful_threshold = 0.5
    sensitive_threshold = 0.5
    temperature = 256
    
    # harmful 和 sensitive 类别
    harmful_category_indices = [0, 1, 2, 3, 4, 5, 6, 7, 9, 12]  # 对应 "Toxic", "Derogatory", "Violent", "Sexual", 等
    sensitive_category_indices = [8, 9, 10, 11, 13, 14]  # 对应 "Public Safety", "Health", "Religion & Belief", 等

    with torch.no_grad():
        for i, (prompts, embeddings, categories) in enumerate(test_loader):
            embeddings, categories = embeddings.to(device), categories.to(device)

            # 模型输出
            outputs = model(embeddings)
            outputs_softmax = torch.sigmoid(outputs)  # 转换为概率

            # outputs_softmax = F.softmax(outputs / temperature, dim=1)

            # 模型预测flag计算
            harmful_scores = outputs_softmax[:, harmful_category_indices]

            sensitive_scores = outputs_softmax[:, sensitive_category_indices]

            predicted_flag = ((harmful_scores > harmful_threshold).any(dim=1) |
                            (sensitive_scores > sensitive_threshold).any(dim=1)).float()

            # 真实flag计算
            harmful_truth_scores = categories[:, harmful_category_indices]
            sensitive_truth_scores = categories[:, sensitive_category_indices]

            true_flag = ((harmful_truth_scores > harmful_threshold).any(dim=1) |
                        (sensitive_truth_scores > sensitive_threshold).any(dim=1)).float()

            # 判断预测和真实的flag是否匹配
            correct_flags = (predicted_flag == true_flag).float().sum().item()

            total_correct += correct_flags
            total_samples += len(categories)

            if i % 100 == 0:
                logging.info(f"Sample {i+1}: Flag Match Accuracy: {correct_flags / len(categories):.4f}")
                
    # 计算最终准确率
    flag_accuracy = total_correct / total_samples
    logging.info(f'Train Overall flag accuracy on the training data: {flag_accuracy:.4f}')

if __name__ == "__main__":
    train_and_save_model()    