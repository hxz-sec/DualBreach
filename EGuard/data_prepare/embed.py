import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import trange
import pandas as pd


model_name = "Llama_3_8b"
layer = -1
data_name = "normal_contents_2k"
col = "prompts"
device = "cuda:2"

# todo:yelp(3,0)
# 检查字段
# 注意advbench_100和dvbench_520用harmful，
# advbench_role_100,advbench_role_5500用prompt
# advbench_role_control_100，suffix，control用prompt_with_adv

tokenizer = AutoTokenizer.from_pretrained(f"/hub/huggingface/models/meta/Llama-3-8B/")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(f"/hub/huggingface/models/meta/Llama-3-8B/")
model = model.to(device)

contents = pd.read_csv(f"/data/home/Xinzhe/GuardBreaker/defend/data/csv/{data_name}.csv")
# contents = pd.read_csv(data_name)
contents = contents[col].tolist()
print(len(contents))

def batch_embed(sentences, batch_size=8):
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



embed_contents = batch_embed(contents, 64)
print(embed_contents.shape)
torch.save(embed_contents, f'/data/home/Xinzhe/GuardBreaker/defend/data/embedding/{model_name}/{layer}/{data_name}_{col}.pt')
