from openai import OpenAI  # for making OpenAI API requests
import os
import json
import itertools
import logging
import time
import pandas as pd
import random
import requests
import argparse

import sys
sys.path.append('/data/home/Xinzhe/GuardBreaker/guard_attack/llm_attack')  
import judge_prompt

def Response_score(args,prompts, api_keys, url, max_attempts=1, timeout=10):    
    attempts = 0
    while attempts < len(api_keys) * max_attempts:
        api_key = random.choice(api_keys)
        client = OpenAI(api_key=api_key, base_url=url)
        
        # 尝试请求
        while True:  # 使用循环来处理重试逻辑
            try:
                response = client.moderations.create(
                    model="text-moderation-stable",
                    input=prompts,
                )
                
                response_data = response.to_dict() if hasattr(response, "to_dict") else response        
                print(response_data)
                time.sleep(10)
                
                if "results" in response_data and response_data["results"]:
                    for i in range(len(prompts)):
                        response_data["results"][i][args.csv_col] = prompts[i]
                    logging.info(response_data)
                    return response_data["results"]
                else:
                    print(response)
                    logging.error("Received an empty or malformed response, trying next API key...")
                    break  # 退出当前请求循环以尝试下一个API键
        
            except Exception  as e:
                logging.error(f"Rate limit exceeded with API key {api_key}: {e}. Waiting for 60 seconds before retrying...")
                api_key = random.choice(api_keys)
                time.sleep(60)  # 等待60秒后再进行下一次请求
            except Exception  as e:
                logging.error(f"Request failed with API key {api_key}: {e}, retrying in 60 seconds...")
                api_key = random.choice(api_keys)
                time.sleep(60)  # 等待60秒后再进行下一次请求
        
        attempts += 1  # 增加尝试次数
        time.sleep(2)  # 可选：在每次尝试之间稍作等待

    logging.info("All attempts failed. No valid response received.")
    return None

def save_responses_to_jsonl(filename, responses):
    with open(filename, 'a') as f:
        for response in responses:
            f.write(json.dumps(response) + '\n')   

def count_jsonl_entries(filename):
    if not os.path.exists(filename):
        return 0
    with open(filename, 'r') as f:
        return sum(1 for _ in f)

def load_prompts_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    return df[args.csv_col].tolist()

def main(args):
    # sirui
    # 初始API密钥和URL


    api_keys_1 = [
        "sk-e9io8l1lreAQv6yX831c2aDe444d4f2b9402730d65C5A3Ac",
        # "sk-PoqlmfodNHPOaugJ115c899186B1456dA9AeE6081bB8782f",
        # "sk-1iecj64toj0ZnnDfFf36E89c79E047D29a31Bc031a156c65",
    ]
    url_1 = "https://api.ai-gaochao.cn/v1"

    # haolun
    api_keys_2 = [
        "sk-DDSQELyZz1cMqMKcbVusqolzPks6JCF7GTBNWptzU3fUPuZS"
    ]
    url_2 = "https://api.chatanywhere.com.cn/v1/moderations"
    # openai
    api_keys_3 = [
        "sk-proj-9dymH5fHVHVVZ690Nv1ST3BlbkFJ7fKa556iLuBsnChlAYQ3",
        "sk-proj-lA5HzS6iL7dhFy5lK2EZT3BlbkFJNy3I41keByjFoUuPhJoS",
    ]
    url_3 = "https://api.openai.com/v1/moderations"



    # 初始化使用第一个API和URL
    current_api_keys = api_keys_1
    current_url = url_1
    batch_size = args.batch_size  # 每次处理prompts数量
    csv_file = args.input_csv_path
    jsonl_file = args.output_jsonl_path
    prompts = load_prompts_from_csv(csv_file)

    responses = []
    fail_count = 0

    for i in range(0, len(prompts), batch_size):  
        batch_prompts = prompts[i:i+batch_size]       
        while fail_count < 4:  # 在达到fail_count限制前一直重试
            response = Response_score(args, batch_prompts, current_api_keys, current_url)
            
            if response is not None:
                responses.extend(response)
                fail_count = 0
                break  # 成功获取响应，跳出当前的内部循环
            else:
                fail_count += 1

            if fail_count >= 4:
                logging.error("Consecutive failures reached limit. Switching to backup API.")
                if current_api_keys == api_keys_1:
                    current_api_keys = api_keys_2
                    current_url = url_2
                    logging.info(f"Switched to API 2")
                if current_api_keys == api_keys_2:
                    current_api_keys = api_keys_3
                    current_url = url_3
                    logging.info(f"Switched to API 3")
                elif current_api_keys == api_keys_3:
                    current_api_keys = api_keys_1
                    current_url = url_1
                    logging.info(f"Switched to API 1")
                fail_count = 0  # 重置fail_count，开始使用备用API继续尝试

        if responses:
            save_responses_to_jsonl(jsonl_file, responses)
            responses = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    file_name = "harmBench_DAN_100"
    input_csv_path = f"/data/home/Xinzhe/GuardBreaker_new/data/csv/{file_name}.csv"
    colmn = judge_prompt.judge_colmn(input_csv_path)
    print("colmn为:")
    print(colmn)

    parser.add_argument('--input_csv_path', type=str, default=input_csv_path)
    parser.add_argument('--csv_col', type=str, default=colmn)
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--output_jsonl_path', type=str, default= f"/data/home/Xinzhe/GuardBreaker_new/data/score/guard_attack/{file_name}_openai.jsonl")
    args = parser.parse_args()

    main(args)


    # 一旦报下面的错误都需要将之前的文件删掉才可以
    # openai.RateLimitError: Error code: 429 - {'error': {'message': '当前分组上游负载已饱和，请稍后再试 
    # (request id: 202411032327512402158209940080)', 'type': 'tokens', 'param': '', 'code': 'rate_limit_exceeded'}}
