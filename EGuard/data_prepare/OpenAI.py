import openai  # for making OpenAI API requests
import os
import json
import itertools
import logging
import time
import pandas as pd
import random
import requests
import argparse


def Response_score(prompts, api_keys, url, max_attempts=2, timeout=10):    
    attempts = 0

    while attempts < len(api_keys) * max_attempts:
        try:
            api_key = random.choice(api_keys)
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            data = {
                "input": prompts  # 传入多个 prompts
            }

            response = requests.post(url, headers=headers, data=json.dumps(data), timeout=timeout)

            # response = requests.post(url, headers=headers, data=json.dumps(data))
            response_data = response.json()
            
            if 'results' in response_data and response_data['results']:
                for i in range(len(prompts)):
                    response_data['results'][i]['prompts'] = prompts[i]
                logging.info(response_data)
                return response_data['results']
            else:
                logging.error("Received an empty or malformed response, trying next API key...")
        
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed with API key {api_key}: {e}, trying next API key...")

        attempts += 1
        time.sleep(2)

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
    # haolun
    api_keys_1 = [
        "sk-DDSQELyZz1cMqMKcbVusqolzPks6JCF7GTBNWptzU3fUPuZS"
    ]
    url_1 = "https://api.chatanywhere.com.cn/v1/moderations"
    # openai
    api_keys_2 = [
        "sk-proj-9dymH5fHVHVVZ690Nv1ST3BlbkFJ7fKa556iLuBsnChlAYQ3",
        "sk-proj-lA5HzS6iL7dhFy5lK2EZT3BlbkFJNy3I41keByjFoUuPhJoS",
    ]
    url_2 = "https://api.openai.com/v1/moderations"
    # sirui
    # 初始API密钥和URL
    api_keys_3 = [
        "sk-Sdi6T5pdnx525EuU7c1c6d5cD4284b3394Cf5dCbE2A6B6A1",
        "sk-7P7LrD18HFHpSeFTC4C238FcE83542F6Ab94C77a735f5aDe"
    ]
    url_3 = "https://api.ai-gaochao.cn/v1/moderations"
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
            response = Response_score(batch_prompts, current_api_keys, current_url)
            
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
    parser.add_argument('--input_csv_path', type=str, default="/data/home/Xinzhe/GAN/data/normal_contents_2k.csv")
    parser.add_argument('--csv_col', type=str, default="prompts")
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--output_jsonl_path', type=str, default="./guardscore/openai-normal-2k.jsonl")
    args = parser.parse_args()

    main(args)
