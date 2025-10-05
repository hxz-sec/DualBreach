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
from torch.optim.lr_scheduler import StepLR,MultiStepLR

# # 设置可见的CUDA设备
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("../log/training_subset_proxy_google.log", mode='w')
    ]
)

# 选择设备
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device: {device}')

# # 定义DNN模型
class DNNClassifier(nn.Module):
    def __init__(self):
        super(DNNClassifier, self).__init__()
        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        # self.fc45 = nn.Linear(64,16)
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
        
        # x = self.dropout(x)
        # x = self.fc45(x)
        
        return x




class TransformerClassifier(nn.Module):
    """ Wrapper of Classifier based on Transformer """
    def __init__(
        self, 
        input_dim = 4096, 
        hidden_dim = 256,
        num_classes = 16,
        num_layers = 3,
        num_heads = 8,
        dropout = 0.1
    ):
        super(TransformerClassifier, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model = input_dim,
                nhead = num_heads,
                dim_feedforward = hidden_dim,
                dropout = dropout
            ),
            num_layers = num_layers
        )
        self.fc = nn.Linear(input_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.transformer(x)
        out = self.dropout(out)
        x = self.fc(x + out)
        
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
    # 初始化模型、损失函数和优化器
    model = DNNClassifier().to(device)
    # model = TransformerClassifier().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size = 10, gamma = 0.1)
    # scheduler = MultiStepLR(optimizer, milestones=[7,16,18], gamma = 0.1)
    criterion_categories = nn.BCEWithLogitsLoss()
    # 训练过程
    num_epochs = 20
    pdist = nn.PairwiseDistance(p=1)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_variance = 0.0
        total_samples = 0
        for i, (prompts, embeddings, categories) in enumerate(train_loader):
            embeddings, categories = embeddings.to(device), categories.to(device)

            outputs = model(embeddings)
            
            loss_categories = criterion_categories(outputs, categories)

            total_variance += pdist(F.sigmoid(outputs), categories).sum().item()
            optimizer.zero_grad()
            loss_categories.backward()
            optimizer.step()

            total_loss += loss_categories.item()
            if i % 100 == 0:
                logging.info(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss_categories.item():.4f}")
            total_samples += len(categories)
        scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        logging.info(f"total: {total_samples}")
        logging.info(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
        total_variance /= total_samples * 2
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], total variance distance: {total_variance:.4f}")
        
    # 保存模型
    torch.save(model.state_dict(), f'../model/{save_model_path}.pth')
    logging.info(f'Model saved to {save_model_path}.pth')

############################################################################################


    print("train dataset evaluate")
    #!! 使用训练集评估模型
    # 评估训练集的acc
    model.eval()
    total_correct = 0
    total_samples = 0
    threshold = 0.8
    # temperature = 256
    
    # harmful 和 sensitive 类别
    # harmful_category_indices = [0, 1, 2, 3, 4, 5, 6, 7, 9, 12]  # 对应 "Toxic", "Derogatory", "Violent", "Sexual", 等
    # sensitive_category_indices = [8, 9, 10, 11, 13, 14]  # 对应 "Public Safety", "Health", "Religion & Belief", 等

    with torch.no_grad():
        for i, (prompts, embeddings, categories) in enumerate(train_loader):
            embeddings, categories = embeddings.to(device), categories.to(device)

            # 模型输出
            outputs = model(embeddings)

            outputs = F.sigmoid(outputs)

            predicted_flag = (outputs > threshold).any(dim=1).float()
            true_flag = (categories > threshold).any(dim=1).float()

            # 判断预测和真实的flag是否匹配
            correct_flags = (predicted_flag == true_flag).float().sum().item()

            total_correct += correct_flags
            total_samples += len(categories)

            if i % 100 == 0:
                logging.info(f"Sample {i+1}: Flag Match Accuracy: {correct_flags / len(categories):.4f}")
                
    # 计算最终准确率
    flag_accuracy = total_correct / total_samples
    logging.info(f'Train Overall flag accuracy on the training data: {flag_accuracy:.4f}')
    
    
def test():
    print("Test dataset evaluate")
    #!! 使用测试集评估模型
    model = DNNClassifier()
    model.load_state_dict(torch.load(f'../model/{save_model_path}.pth'))
    model.to(device)
    model.eval()
    total_correct = 0
    total_samples = 0
    threshold = 0.5
    # temperature = 256

    pdist = nn.PairwiseDistance(p=1)
    total_variance = 0.0
    
    with torch.no_grad():
        for i, (prompts, embeddings, categories) in enumerate(test_loader):
            embeddings, categories = embeddings.to(device), categories.to(device)

            # 模型输出
            outputs = model(embeddings)
            outputs = F.sigmoid(outputs)  # 转换为概率

            predicted_flag = (outputs > threshold).any(dim=1).float()
            true_flag = (categories > threshold).any(dim=1).float()
            
            similarity_score = pdist(outputs,categories)
            total_variance += similarity_score.sum().item()

            # 判断预测和真实的flag是否匹配
            correct_flags = (predicted_flag == true_flag).float().sum().item()

            total_correct += correct_flags
            total_samples += len(categories)

            if i % 100 == 0:
                logging.info(f"Sample {i+1}: Flag Match Accuracy: {correct_flags / len(categories):.4f}")

    # 计算最终准确率
    flag_accuracy = total_correct / total_samples
    total_variance /= total_samples * 16
    logging.info(f'Test Overall flag accuracy on the test data: {flag_accuracy:.4f}')
    logging.info(f"total variance distance: {total_variance:.4f}")
    # logging.info(f"pairwise distance: {pdistance:.4f}")
    


if __name__ == "__main__":


    file_path = "/data/home/Xinzhe/GuardBreaker_new/github/proxy_model/data"
    data_paths = {
        "harmful": {
            "embedding": f'{file_path}/harmful_subset_google_800_prompts.pt',
            "csv": f'{file_path}/harmful_subset_google_800.csv',
            "jsonl": f'{file_path}/harmful_subset_google_800_google.jsonl'
        },
        "normal": {
            "embedding": f'{file_path}/normal_subset_google_800_prompts.pt',
            "csv": f'{file_path}/normal_subset_google_800.csv',
            "jsonl": f'{file_path}/normal_subset_google_800_google.jsonl'
        },
    }



    all_prompts = []
    all_embeddings = []
    all_categories = []

    for key, paths in data_paths.items():

        

        embedding_data = torch.load(paths['embedding'])

        prompts = pd.read_csv(paths['csv'])["prompts"].tolist()

        categories = load_jsonl_data(paths['jsonl'])

        all_prompts.extend(prompts)
        all_embeddings.append(embedding_data)
        all_categories.extend(categories)


    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_categories = torch.stack(all_categories)


    dataset = PromptDataset(all_prompts, all_embeddings, all_categories)


    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


    logging.info(f"Total prompts count: {len(all_prompts)}")
    logging.info(f"Training dataset size: {train_size}")
    logging.info(f"Testing dataset size: {test_size}")

    batch_size = 1


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    save_model_path = "subset_proxy_google_test"
    
    train_and_save_model()
    test()