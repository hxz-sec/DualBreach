import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd
import logging
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR,MultiStepLR

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    handlers=[
        logging.StreamHandler(), 
        logging.FileHandler("../log/proxy_openai_test.log", mode='w')  
    ]
)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device: {device}')

# 定义DNN模型
class DNNClassifier(nn.Module):
    def __init__(self):
        super(DNNClassifier, self).__init__()
        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        # self.fc4 = nn.Linear(64, 16)
        # self.fc5 = nn.Linear(16, 11)  # 11 categories corresponding to OpenAI Moderation API
        self.fc45 = nn.Linear(64, 11)  # 11 categories corresponding to OpenAI Moderation API
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        # x = self.relu(self.fc4(x))
        # x = self.dropout(x)
        # x = self.fc5(x)
        
        x = self.dropout(x)
        x = self.fc45(x)
        return x


# 创建一个包含 (prompt, embedding, label，categories，category_scores) 的数据集
class PromptDataset(torch.utils.data.Dataset):
    def __init__(self, prompts, embeddings, labels, categories, category_scores):
        self.prompts = prompts
        self.embeddings = embeddings
        self.labels = labels
        self.categories = categories  # 新增categories信息
        self.category_scores = category_scores  # 修改为category_scores信息

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.prompts[idx], self.embeddings[idx], self.labels[idx], self.categories[idx], self.category_scores[idx]


def load_jsonl_data(jsonl_file):
    categories = []
    category_scores = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            category_values = [int(data['categories'][key]) for key in data['categories']]
            score_values = list(data['category_scores'].values())
            categories.append(torch.tensor(category_values, dtype=torch.float))
            category_scores.append(torch.tensor(score_values, dtype=torch.float))
    return categories, category_scores

def generate_json_result(prompt, outputs, predicted_categories):
    categories_dict = {
        "sexual": bool(predicted_categories[0][0].item()),
        "hate": bool(predicted_categories[0][1].item()),
        "harassment": bool(predicted_categories[0][2].item()),
        "self-harm": bool(predicted_categories[0][3].item()),
        "sexual/minors": bool(predicted_categories[0][4].item()),
        "hate/threatening": bool(predicted_categories[0][5].item()),
        "violence/graphic": bool(predicted_categories[0][6].item()),
        "self-harm/intent": bool(predicted_categories[0][7].item()),
        "self-harm/instructions": bool(predicted_categories[0][8].item()),
        "harassment/threatening": bool(predicted_categories[0][9].item()),
        "violence": bool(predicted_categories[0][10].item())
    }

    category_scores_dict = {
        "sexual": outputs[0][0].item(),
        "hate": outputs[0][1].item(),
        "harassment": outputs[0][2].item(),
        "self-harm": outputs[0][3].item(),
        "sexual/minors": outputs[0][4].item(),
        "hate/threatening": outputs[0][5].item(),
        "violence/graphic": outputs[0][6].item(),
        "self-harm/intent": outputs[0][7].item(),
        "self-harm/instructions": outputs[0][8].item(),
        "harassment/threatening": outputs[0][9].item(),
        "violence": outputs[0][10].item()
    }

    json_result = {
        "flagged": predicted_categories.any().item(),
        "categories": categories_dict,
        "category_scores": category_scores_dict,
        "prompt": prompt
    }

    return json_result


def train_and_save_model():

    # 定义每个数据集要加载的数据数量,最终训练数据量
    data_limits = {
        "harmful": 75898,
        "trivia_qa_train": 35000,
        "openbookqa": 5000,
        "filtered_wiki_qa": 898,
        "yelp_train": 35000
    }


    # 定义每个数据集要加载的数据数量,测试
    # data_limits = {
    #     "harmful": 800,
    #     "trivia_qa_train": 200,
    #     "openbookqa": 200,
    #     "filtered_wiki_qa": 200,
    #     "yelp_train": 200
    # }

    # 定义数据集路径
    data_paths = {
        "harmful": {
            "embedding": '../data/embedding/Llama-3-8b/harmful_contents_prompts.pt',
            "csv": '../data/csv/harmful_contents.csv',
            "jsonl": '../data/score/harmful_contents_75000_openai.jsonl'
        },
        "trivia_qa_train": {
            "embedding": '../data/embedding/Llama-3-8b/trivia_qa_train_prompts.pt',
            "csv": '../data/csv/trivia_qa_train.csv',
            "jsonl": '../data/score/trivia_qa_train_35000_openai.jsonl'
        },
        "openbookqa": {
            "embedding": '../data/embedding/Llama-3-8b/openbookqa_prompts.pt',
            "csv": '../data/csv/openbookqa.csv',
            "jsonl": '../data/score/openbookqa_5000_openai.jsonl'
        },
        "filtered_wiki_qa": {
            "embedding": '../data/embedding/Llama-3-8b/filtered_wiki_qa_prompts.pt',
            "csv": '../data/csv/filtered_wiki_qa.csv',
            "jsonl": '../data/score/filtered_wiki_qa_1000_openai.jsonl'
        },
        "yelp_train": {
            "embedding": '../data/embedding/Llama-3-8b/yelp_train_prompts.pt',
            "csv": '../data/csv/yelp_train.csv',
            "jsonl": '../data/score/yelp_train_35000_openai.jsonl'
        }
    }

    # 加载和合并数据
    all_prompts = []
    all_embeddings = []
    all_labels = []
    all_categories = []
    all_category_scores = []

    harmful_counts = {}
    normal_total_counts = {"embedding": 0, "prompts": 0, "categories": 0, "category_scores": 0}

    for key, paths in data_paths.items():
        embedding_data = torch.load(paths['embedding'])[:data_limits[key]]
        prompts = pd.read_csv(paths['csv'])["prompts"].tolist()[:data_limits[key]]
        categories, category_scores = load_jsonl_data(paths['jsonl'])
        categories = categories[:data_limits[key]]
        category_scores = category_scores[:data_limits[key]]

        if key == "harmful":
            labels = torch.ones(len(prompts), dtype=torch.long)
            harmful_counts = {
                "embedding": len(embedding_data),
                "prompts": len(prompts),
                "categories": len(categories),
                "category_scores": len(category_scores)
            }
        else:
            labels = torch.zeros(len(prompts), dtype=torch.long)
            normal_total_counts["embedding"] += len(embedding_data)
            normal_total_counts["prompts"] += len(prompts)
            normal_total_counts["categories"] += len(categories)
            normal_total_counts["category_scores"] += len(category_scores)

        all_prompts.extend(prompts)
        all_embeddings.append(embedding_data)
        all_labels.append(labels)
        all_categories.extend(categories)
        all_category_scores.extend(category_scores)

    # 合并所有的embedding和labels
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    dataset = PromptDataset(all_prompts, all_embeddings, all_labels, all_categories, all_category_scores)

    # 打印 harmful 和 normal 数据集的数量信息
    logging.info("Harmful dataset counts:")
    for key, count in harmful_counts.items():
        logging.info(f"  {key}: {count}")

    logging.info("Normal datasets combined counts:")
    for key, count in normal_total_counts.items():
        logging.info(f"  {key}: {count}")

        # 检查 harmful 和 normal 数据集的总数量是否一致
        if harmful_counts[key] != count:
            logging.warning(f"Mismatch in '{key}' between harmful and combined normal datasets: {harmful_counts[key]} vs {count}")
        else:
            logging.info(f"Counts for '{key}' match between harmful and combined normal datasets.")

    # 按照 8-2 划分数据集为训练集和测试集，并进行打乱
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 打印训练集和测试集的大小
    logging.info(f"Total prompts count: {len(all_prompts)}")
    logging.info(f"Training dataset size: {train_size}")
    logging.info(f"Testing dataset size: {test_size}")

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 初始化模型、损失函数和优化器
    model = DNNClassifier().to(device)
    model.load_state_dict(torch.load("../model/proxy_openai.pth"))

    # 使用测试集评估模型
    model.eval()
    correct = 0
    total = 0
    incorrect_total = 0
    match_amount = 0


    with torch.no_grad():
        for i, (prompts, embeddings, targets, categories, category_scores) in enumerate(test_loader):
            # 确保所有张量都在同一设备上
            embeddings, targets = embeddings.to(device), targets.to(device)
            categories, category_scores = categories.to(device), category_scores.to(device)
            
            outputs = model(embeddings)
            # outputs_softmax = F.softmax(outputs / temperature)
            outputs = F.sigmoid(outputs)
            predicted_categories = (outputs > 0.5).float()
            
                        
            # 计算每个标签的匹配情况
            correct_labels = (predicted_categories == categories).float().sum().item()


            # 检查是否完全匹配
            # is_correct = torch.all(predicted_categories == categories).item()
            # is_correct = (predicted_categories == categories).all().item()


            # 计算最终的flag状态
            predicted_flagged = predicted_categories.any().item()
            target_flagged = categories.any().item()
            
            # 允许一定比例的标签预测错误，且flagged状态必须匹配
            threshold = 1  # 80%的标签匹配就算正确
            is_correct = (correct_labels / len(categories[0]) >= threshold) and (predicted_flagged == target_flagged)

            if predicted_flagged == target_flagged & predicted_flagged == targets:
                match_amount += 1

            if is_correct:
                correct += 1
                logging.info(f"Sample {i+1}: Correct prediction")
            else:
                incorrect_total += 1
                logging.info(f"Sample {i+1}: Incorrect prediction")

            # 打印每个样本的预测结果
            # json_result = generate_json_result(prompts[0], outputs, predicted_categories)
            # logging.info(json.dumps(json_result, indent=2))

            total += 1

    accuracy = correct / total if total > 0 else 0
    logging.info(f'Match data : {match_amount:.4f}')
    logging.info(f'Accuracy of the model on the test data: {accuracy:.4f}')
    logging.info(f'Total correct predictions: {correct}')
    logging.info(f'Total incorrect predictions: {incorrect_total}')
    logging.info(f'Total samples in the test set: {total}')

if __name__ == "__main__":
    train_and_save_model()
