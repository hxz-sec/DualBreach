#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import time
import wandb
import argparse
import random
import sys
sys.path.insert(0, './GPT2ForwardBackward')

# from nltk import data

# data.path.append("/data/home/Xinzhe/COLD-Attack")

from nltk.corpus import stopwords
from opt_util import *
from util import *
from bleuloss import batch_log_bleulosscnn_ae
import torch.nn as nn


stop_words = set(stopwords.words('english'))


# class DNNClassifier(nn.Module):
#     def __init__(self):
#         super(DNNClassifier, self).__init__()
#         self.fc1 = nn.Linear(4096, 1024)
#         self.fc2 = nn.Linear(1024, 256)
#         self.fc3 = nn.Linear(256, 64)
#         # self.fc4 = nn.Linear(64, 16)
#         # self.fc5 = nn.Linear(16, 11)  # 11 categories corresponding to OpenAI Moderation API
#         self.fc45 = nn.Linear(64, 11)  # 11 categories corresponding to OpenAI Moderation API
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(p=0.5)
        
#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = self.relu(self.fc3(x))
#         # x = self.relu(self.fc4(x))
#         # x = self.dropout(x)
#         # x = self.fc5(x)
        
#         x = self.dropout(x)
#         x = self.fc45(x)
#         return x


class DNNClassifier(nn.Module):
    def __init__(self):
        super(DNNClassifier, self).__init__()
        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
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


def options():
    parser = argparse.ArgumentParser()
    ## setting
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--no-cuda", action="store_true", help="no cuda")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--print-every", type=int, default=200)
    parser.add_argument("--pretrained_model", type=str, default="llama2")
    parser.add_argument("--pretrained_judge_model", type=str, default="Qwen2.5-7B-Instruct")
    parser.add_argument("--pretrained_guard_model", type = str, default = "None")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--straight-through", action="store_true")
    parser.add_argument("--topk", type=int, default=0)
    parser.add_argument("--rl-topk", type=int, default=0)
    parser.add_argument("--lexical", type=str, default='max', choices=['max', 'ppl_max', 'all', 'bleu'])
    parser.add_argument("--lexical-variants", action="store_true", help="")
    parser.add_argument("--if-zx", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--guard", action="store_true")
    parser.add_argument("--device-id", type = int, default = 0, help = "device id for generating adversarial examples")
    parser.add_argument('--guard-device-id', type=int, default=1, help='device id for guarding model')
    ## experiment
    parser.add_argument("--input-file", type=str,
                        default="./data/lexical/commongen_data/test.multi.constraint.json")
    parser.add_argument("--version", type=str, default="")
    parser.add_argument("--start", type=int, default=1, help="loading data from ith examples.")
    parser.add_argument("--end", type=int, default=10, help="loading data util ith examples.")
    parser.add_argument("--repeat-batch", type=int, default=1, help="loading data util ith examples.")
    parser.add_argument("--mode", type=str, default='constrained_langevin',
                        choices=['suffix', 'control', 'paraphrase'])
    parser.add_argument("--control-type", type=str, default='sentiment', choices=['sentiment', 'lexical', 'style', 'format'])
    ## model
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--length", type=int, default=15, help="maximum length of optimized logits.")
    parser.add_argument("--max-length", type=int, default=50, help="maximum length of complete sentence.")
    parser.add_argument("--frozen-length", type=int, default=0, help="length of optimization window in sequence.")
    parser.add_argument("--goal-weight", type=float, default=0.1)
    parser.add_argument("--rej-weight", type=float, default=0.05)
    parser.add_argument("--abductive-filterx", action="store_true", help="filter out keywords included in x")
    parser.add_argument("--lr-nll-portion", type=float, default=1)
    parser.add_argument("--prefix-length", type=int, default=0, help="length of prefix.")
    parser.add_argument("--counterfactual-max-ngram", type=int, default=3)
    parser.add_argument("--no-loss-rerank", action="store_true", help="")
    parser.add_argument("--use-sysprompt", action="store_true", help="")
    # temperature
    parser.add_argument("--input-lgt-temp", type=float, default=1,
                        help="temperature of logits used for model input.")
    parser.add_argument("--output-lgt-temp", type=float, default=1,
                        help="temperature of logits used for model output.")
    parser.add_argument("--rl-output-lgt-temp", type=float, default=1,
                        help="temperature of logits used for model output.")
    parser.add_argument("--init-temp", type=float, default=0.1,
                        help="temperature of logits used in the initialization pass. High => uniform init.")
    parser.add_argument("--init-mode", type=str, default='original', choices=['random', 'original'])
    # lr
    parser.add_argument("--stepsize", type=float, default=0.1, help="learning rate in the backward pass.")
    parser.add_argument("--stepsize-ratio", type=float, default=1, help="")
    parser.add_argument("--stepsize-iters", type=int, default=1000, help="")
    # iterations
    parser.add_argument("--num-iters", type=int, default=1000)
    parser.add_argument("--min-iters", type=int, default=0, help="record best only after N iterations")
    parser.add_argument("--noise-iters", type=int, default=1, help="add noise at every N iterations")
    parser.add_argument("--optim-y-iters", type=int, default=5, help="optimize y at every N iterations")
    parser.add_argument("--paraphrase-iters", type=int, default=50, help="paraphrase at every N iterations")
    parser.add_argument("--win-anneal-iters", type=int, default=-1, help="froze the optimization window after N iters")
    parser.add_argument("--constraint-iters", type=int, default=1000,
                        help="add one more group of constraints from N iters")
    # gaussian noise
    parser.add_argument("--gs_mean", type=float, default=0.0)
    parser.add_argument("--gs_std", type=float, default=0.01)
    parser.add_argument("--large-noise-iters", type=str, default="-1", help="Example: '50,1000'")
    parser.add_argument("--large_gs_std", type=str, default="1", help="Example: '1,0.1'")

    args = parser.parse_args()
    return args

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    args = options()
    # device_ids = torch.cuda.device_count()
    generate_device_id = args.device_id
    guard_device_id = args.guard_device_id
    
    if args.pretrained_guard_model != "None":
        assert generate_device_id != guard_device_id, "generate_device_id and guard_device_id should be different, while we got generate_device_id = {}, guard_device_id = {}".format(generate_device_id, guard_device_id)
    else:
        guard_device_id = -1 # set -1 as unuse

    device = "cuda:{}".format(generate_device_id) if torch.cuda.is_available() and not args.no_cuda else "cpu"
    guard_model_device = "cuda:{}".format(guard_device_id) if torch.cuda.is_available() and not args.no_cuda else "cpu"
    
    model_path_dicts = {"Llama-2-7b-chat-hf": "/hub/huggingface/models/meta/llama2-hf/llama-2-7b-chat-hf",
                        "Llama-3-8b" : "/hub/huggingface/models/meta/Llama-3-8B",
                        "Llama-3-8b-instruct" : "/hub/huggingface/models/meta/Llama-3-8B-Instruct",
                        "Vicuna-7b-v1.5": "lmsys/vicuna-7b-v1.5",
                        "guanaco-7b": "TheBloke/guanaco-7B-HF",
                        "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2", 
                        "Llama-Guard-3-8B":"/hub/huggingface/models/meta/Llama-Guard-3-8B",
                        "gemma2" : "/hub/huggingface/models/CLAS24/gemma-2-9b-it/",
                        }
    guard_model_path_dicts = {
            "llama-guard-3-8b" : "/hub/huggingface/models/meta/Llama-Guard-3-8B",
            }
    model_path = model_path_dicts[args.pretrained_model]
    if args.pretrained_guard_model in guard_model_path_dicts:
        guard_model_path = guard_model_path_dicts[args.pretrained_guard_model]
    else:
        guard_model_path = ""
    if args.seed != -1:
        seed_everything(args.seed)
    # Load pretrained model
    model, tokenizer = load_model_and_tokenizer(model_path,
                                                low_cpu_mem_usage=True,
                                                use_cache=False,
                                                device=device)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    if guard_model_path != "":
        guard_model, guard_tokenizer, guard_model_embedding_weights = load_guard_model_and_tokenizer(guard_model_path, low_cpu_mem_usage = True, use_cache = False, device = guard_model_device)
    elif args.pretrained_guard_model != "None":
        guard_model = DNNClassifier()
        guard_model.load_state_dict(torch.load(f"/data/home/Xinzhe/GuardBreaker/proxy_model/model/{args.pretrained_guard_model}.pth"))
        guard_model.to(guard_device_id)
        guard_model.eval()
        guard_tokenizer = AutoTokenizer.from_pretrained(
            "/hub/huggingface/models/meta/Llama-3-8B",
            trust_remote_code=True,
            use_fast=False,
            use_cache=True,
        )
        if not guard_tokenizer.pad_token:
            guard_tokenizer.pad_token = guard_tokenizer.eos_token
        guard_model_embedding_weights = None
    else:
        guard_model, guard_tokenizer, guard_model_embedding_weights = None, None, None

    # Freeze GPT-2 weights
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
        
    if "suffix" in args.mode:
        from attack_suffix import attack_generation
        print("suffix")
        attack_generation(model, tokenizer, device, args, guard_model = guard_model, guard_tokenizer = guard_tokenizer, guard_model_embedding_weights = guard_model_embedding_weights, guard_model_device = guard_model_device)
    elif "paraphrase" in args.mode:
        print("paraphrase")
        from attack_paraphrase import attack_generation
        attack_generation(model, tokenizer, device, args, guard_model = guard_model, guard_tokenizer = guard_tokenizer, guard_model_embedding_weights = guard_model_embedding_weights, guard_model_device = guard_model_device)
    elif "control" in args.mode:
        print("control")
        from attack_control import attack_generation
        attack_generation(model, tokenizer, device, args, guard_model = guard_model, guard_tokenizer = guard_tokenizer, guard_model_embedding_weights = guard_model_embedding_weights, guard_model_device = guard_model_device)

if __name__ == "__main__":
    main()
