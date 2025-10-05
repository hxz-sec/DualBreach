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
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    handlers=[
        logging.StreamHandler(), 
        logging.FileHandler("../log/proxy_openai_evaluate.log", mode='w')  
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
    def __init__(self, prompts, embeddings, labels,categories, category_scores):
        self.prompts = prompts
        self.embeddings = embeddings
        self.labels = labels
        self.categories = categories
        self.category_scores = category_scores

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.prompts[idx], self.embeddings[idx], self.labels[idx], self.categories[idx], self.category_scores[idx]

def generate_json_result(prompt, outputs, predicted_categories):
    predicted_categories = predicted_categories.tolist()  # 将张量转换为标量列表
    categories_dict = {
        "sexual": bool(predicted_categories[0]),
        "hate": bool(predicted_categories[1]),
        "harassment": bool(predicted_categories[2]),
        "self-harm": bool(predicted_categories[3]),
        "sexual/minors": bool(predicted_categories[4]),
        "hate/threatening": bool(predicted_categories[5]),
        "violence/graphic": bool(predicted_categories[6]),
        "self-harm/intent": bool(predicted_categories[7]),
        "self-harm/instructions": bool(predicted_categories[8]),
        "harassment/threatening": bool(predicted_categories[9]),
        "violence": bool(predicted_categories[10])
    }

    category_scores_dict = {
        "sexual": format(outputs[0][0].item(), ".15f"),
        "hate": format(outputs[0][1].item(), ".15f"),
        "harassment": format(outputs[0][2].item(), ".15f"),
        "self-harm": format(outputs[0][3].item(), ".15f"),
        "sexual/minors": format(outputs[0][4].item(), ".15f"),
        "hate/threatening": format(outputs[0][5].item(), ".15f"),
        "violence/graphic": format(outputs[0][6].item(), ".15f"),
        "self-harm/intent": format(outputs[0][7].item(), ".15f"),
        "self-harm/instructions": format(outputs[0][8].item(), ".15f"),
        "harassment/threatening": format(outputs[0][9].item(), ".15f"),
        "violence": format(outputs[0][10].item(), ".15f")
    }


    json_result = {
        "flagged": any(predicted_categories),  # 检查是否有任意类别被标记
        "categories": categories_dict,
        "category_scores": category_scores_dict,
        "prompt": prompt
    }

    return json_result


def load_jsonl_data(jsonl_file):
    categories = []
    category_scores = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            category_values = [int(data['categories'][key]) for key in data['categories']] # 0 or 1, i.e., False or True
            score_values = list(data['category_scores'].values()) # prob from 0.0 to 1.0
            categories.append(torch.tensor(category_values, dtype=torch.float))
            category_scores.append(torch.tensor(score_values, dtype=torch.float))
    return categories, category_scores

def train_and_save_model():

    file = "harmBench_role_suffix_100_best"

    if file == "advbench_100" or file == "advbench_520" or file in ["DNA_100","DNA_939","harmBench_100","harmBench_320"]:
        colmn = "harmful"
        embedding_file = f'/data/home/Xinzhe/GuardBreaker/guard_attack/data/embedding/Llama_3_8b/-1/{file}_{colmn}.pt'  # 正确的字符串路径
        csv_file = f'/data/home/Xinzhe/GuardBreaker/guard_attack/data/csv/{file}.csv'   
        jsonl_file = f"/data/home/Xinzhe/GuardBreaker/guard_attack/data/score/{file}_openai.jsonl" 

    elif file == "advbench_role_100" or file == "advbench_role_5500" or file in ["DNA_role_100","DNA_role_5500","harmBench_role_100","harmBench_role_5500"]:
        colmn = "prompt"
        embedding_file = f'/data/home/Xinzhe/GuardBreaker/guard_attack/data/embedding/Llama_3_8b/-1/{file}_{colmn}.pt'  # 正确的字符串路径
        csv_file = f'/data/home/Xinzhe/GuardBreaker/guard_attack/data/csv/{file}.csv'
        jsonl_file = f"/data/home/Xinzhe/GuardBreaker/guard_attack/data/score/{file}_openai.jsonl" 

    elif 'best' in file:    
        colmn = "prompt_with_adv"
        embedding_file = f'/data/home/Xinzhe/GuardBreaker/guard_attack/data/embedding/Llama_3_8b/-1/{file}_{colmn}.pt'  # 正确的字符串路径
        csv_file = f'/data/home/Xinzhe/GuardBreaker/guard_attack/data/csv/{file}.csv'
        jsonl_file = f"/data/home/Xinzhe/GuardBreaker/guard_attack/data/score/{file}_openai.jsonl" 
    elif file == "harmful_contents":
        colmn = "prompts"
        embedding_file = f'/data/home/Xinzhe/GuardBreaker/proxy_model/data/embedding/Llama-3-8b/{file}_{colmn}.pt'  # 正确的字符串路径
        csv_file = f'/data/home/Xinzhe/GuardBreaker/proxy_model/data/csv/{file}.csv'      
        jsonl_file = f"/data/home/Xinzhe/GuardBreaker/guard_attack/data/score/{file}_openai.jsonl"
    elif "prp" in file:
        colmn = "prompt_with_adv"
        embedding_file = f'/data/home/Xinzhe/GuardBreaker/baseline/PRP/embedding/Llama_3_8b/-1/{file}_{colmn}.pt'
        csv_file = f'/data/home/Xinzhe/GuardBreaker/baseline/PRP/data/{file}.csv'
        jsonl_file = f'/data/home/Xinzhe/GuardBreaker/baseline/PRP/score/{file}_openai.jsonl'

    output_file = f'/data/home/Xinzhe/GuardBreaker/baseline/PRP/score/{file}_proxy_openai.jsonl'


    embedding_data = torch.load(embedding_file)  # 从字符串路径加载数据
    prompts = pd.read_csv(csv_file)[colmn].tolist()
    labels = torch.ones(len(prompts), dtype=torch.long)
    categories, category_scores = load_jsonl_data(jsonl_file)

    dataset = PromptDataset(prompts, embedding_data, labels, categories,category_scores)

    test_loader = DataLoader(dataset, batch_size=1)

    model = DNNClassifier().to(device)
    model.load_state_dict(torch.load("../model/proxy_openai.pth"))
    print("ini over")
    model.eval()


    match_count = 0
    success_count = 0
    total = 0
    fail_count = 0
    unequal_cnt = 0
    TT_cnt = 0
    TF_cnt = 0
    FT_cnt = 0
    FF_cnt = 0


    with open(output_file, 'a') as jsonl_file:  # 使用文件句柄而不是字符串
        with torch.no_grad():
            for i, (prompts, embeddings, label, categories,category_scores) in tqdm(enumerate(test_loader), desc="Evaluating"):
                
                total += 1

                # 确保所有张量都在同一设备上
                # prompts = prompts.to(device)
                embeddings = embeddings.to(device)
                label = label.to(device)
                categories = categories.to(device)
                category_scores = category_scores.to(device)

                # print("label: ", label)
                # print("label.item(): ", label.item())
                # print("label == True: ", (label == True).item())
                
                outputs = model(embeddings)
                # outputs_softmax = F.softmax(outputs / temperature)
                outputs = F.sigmoid(outputs)
                predicted_categories = (outputs > 0.5).float()


                # 计算当前的flag状态
                predicted_flagged = predicted_categories.any().item()
                # print("predicted_flagged", predicted_flagged)
                
                # get the index of the predicted_flagged that has value 1
                # predicted_flagged_index = torch.where(predicted_categories == 1)[0]
                # print("predicted_flagged_index", predicted_flagged_index)
                
                # 计算预期的flag状态
                target_flagged = categories.any().item()     
                # print("target_flagged", target_flagged)

                # 生成结果
                json_result = generate_json_result(prompts[0], outputs, predicted_categories.squeeze())  # 修复此处为 prompts[0]

                # 写入到jsonl文件中
                json.dump(json_result, jsonl_file)
                jsonl_file.write('\n')

                # if predicted_flagged != target_flagged:
                #     # print("ERROR!!!!!!!!!")
                #     unequal_cnt += 1
                # else:

                # if predicted_flagged == label.item():
                #     # 代理模型识别成功
                #     success_count += 1
                # else:
                #     fail_count += 1   

                # if (predicted_flagged == target_flagged) and (int(predicted_flagged) == label.item()):
                #     # 完全预测成功
                #     success_count += 1
                # elif (predicted_flagged == target_flagged) and (int(predicted_flagged) != label.item()):
                #     # 与原始数据预测成功，但是与实际label不一致
                #     match_count += 1
                # else:
                #     fail_count += 1
                    
                if int(predicted_flagged) == label.item(): 
                    # 代理模型识别为有害样本
                    if predicted_flagged == target_flagged:
                        # 如果这里成功说明代理模型预测样本与测试集经过OpenAI API预测的结果一致
                        TT_cnt += 1
                    else:
                        TF_cnt += 1
                else:
                    # 代理模型识别为无害样本
                    if predicted_flagged == target_flagged:
                        # 如果这里成功说明代理模型预测样本与测试集经过OpenAI API预测的结果一致
                        FT_cnt += 1
                    else:
                        FF_cnt += 1

    accuracy_success = success_count / total if total > 0 else 0
    accuracy_match = match_count / total if total > 0 else 0
    accuracy_fail = fail_count / total if total > 0 else 0

    # logging.info(f'Accuracy of the model on the test data: {accuracy:.4f}')
    
    print("代理模型认为有害而且OpenAI API预测为有害: ", TT_cnt)
    print("代理模型认为有害但是OpenAI API预测为无害: ", TF_cnt)
    print("代理模型认为无害而且OpenAI API预测为无害: ", FT_cnt)
    print("代理模型认为无害但是OpenAI API预测为有害: ", FF_cnt)
    
    print("代理模型认为有害的数量: ", TT_cnt + TF_cnt)
    print("代理模型认为有害的比例: ", (TT_cnt + TF_cnt) / total)
    print("OpenAI API预测为有害的数量: ", TT_cnt + FF_cnt)
    print("OpenAI API预测为无害的数量: ", TF_cnt + FT_cnt)
    
    # logging.info(f'数据集名称: {file}')
    # logging.info(f'数据集数量为: {total}')
    # logging.info(f'三者识别完全匹配数量: {success_count}')
    # logging.info(f'三者识别完全匹配识别成功率: {accuracy_success}')

    # logging.info(f'预测相同但与label不一致数量: {match_count}')
    # logging.info(f'预测相同但与label不一致比例: {accuracy_match}')
    # # logging.info(f'两者匹配但与label不匹配的数量: {accuracy_success}')
    
    # logging.info(f'匹配不成功数量: {fail_count}')
    # logging.info(f'两者匹配但与label不匹配的成功率: {accuracy_fail}')    

if __name__ == "__main__":
    train_and_save_model()
