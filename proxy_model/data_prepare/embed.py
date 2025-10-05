import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import trange
import pandas as pd
import judge_prompt


model_name = "Llama_3_8b"
layer = -1
data_name = "harmful_subset_openai_500"
device = "cuda:1"


tokenizer = AutoTokenizer.from_pretrained(f"/hub/huggingface/models/meta/Llama-3-8B/")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(f"/hub/huggingface/models/meta/Llama-3-8B/")
model = model.to(device)

# 加载数据
contents = pd.read_csv(f"/data/home/Xinzhe/GuardBreaker_new/data/csv/{data_name}.csv")
# col = judge_prompt.judge_colmn(f"/data/home/Xinzhe/GuardBreaker_new/data/csv/{data_name}.csv")
col = "prompts"
contents = contents[col].tolist()

# 限制 contents 的长度为不超过 50,000
# contents = contents[:max_length]

print(f"总共 {len(contents)} 条数据将用于嵌入。")

def batch_embed(sentences, batch_size=32):
    all_embedding_outputs = []
    for i in trange(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden_state = outputs.hidden_states[layer]
        embedding = hidden_state.mean(dim=1)
        all_embedding_outputs.append(embedding.cpu())
    return torch.cat(all_embedding_outputs, dim=0)

embed_contents = batch_embed(contents, batch_size=4)
print(f"嵌入完成，结果形状为：{embed_contents.shape}")

# 保存嵌入结果
output_path = f'/data/home/Xinzhe/GuardBreaker_new/data/embedding/Llama_3_8b/-1/{data_name}_{col}.pt'
torch.save(embed_contents, output_path)
print(f"嵌入结果已保存到 {output_path}")
