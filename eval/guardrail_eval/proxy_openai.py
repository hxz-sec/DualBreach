import torch
import torch.nn as nn
import pandas as pd
from tqdm import trange
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

import sys
sys.path.append('/data/home/Xinzhe/GuardBreaker/guard_attack/llm_attack')  
import judge_prompt


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


proxy_model = "openai"
file_name = "harmBench_100"
input_csv_path = f"/data/home/Xinzhe/GuardBreaker_new/data/csv/{file_name}.csv"
model_path = f"/data/home/Xinzhe/GuardBreaker_new/proxy_model/model/kmeans_proxy_{proxy_model}_test.pth"
device = "cuda:2"
colmn = judge_prompt.judge_colmn(input_csv_path)

tokenizer = AutoTokenizer.from_pretrained(f"/hub/huggingface/models/meta/Llama-3-8B/")
tokenizer.pad_token = tokenizer.eos_token
embed_model = AutoModelForCausalLM.from_pretrained(f"/hub/huggingface/models/meta/Llama-3-8B/")
embed_model = embed_model.to(device)

model = DNNClassifier().eval()
model.load_state_dict(torch.load(model_path))
model.to(device)

df = pd.read_csv(input_csv_path)

safe_label_count = 0
total_count = 0

def batch_embed(sentences, batch_size=32):
    all_embedding_outputs = []
    for i in trange(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = embed_model(**inputs, output_hidden_states=True)
        hidden_state = outputs.hidden_states[-1]
        embedding = hidden_state.mean(dim=1)
        all_embedding_outputs.append(embedding.cpu())
    return torch.cat(all_embedding_outputs, dim=0)

embed_contents = batch_embed(df[colmn].tolist(), batch_size=4)

for i in trange(0, len(embed_contents)):
    total_count += 1
    embedding = embed_contents[i].to(device)
    outputs = model(embedding)
    outputs = F.sigmoid(outputs)
    predicted_categories = (outputs > 0.5).float()
    predicted_flagged = predicted_categories.any().item()
    safe_label_count += 1 - predicted_flagged
    
    
print("file_name为:",file_name)
print("colmn为:")
print(f"总共{total_count}个样本")
print(f"安全个数：{safe_label_count} ({safe_label_count/total_count * 100}%)")