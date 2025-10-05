import requests
import json
import pandas as pd
from tqdm import trange
import time
import argparse
import os
import csv

import sys
sys.path.append('/data/home/Xinzhe/GuardBreaker/guard_attack/llm_attack')  
import judge_prompt

# 增加字段大小限制
csv.field_size_limit(10 * 1024 * 1024)  # 设置为10MB

def prepare_text(text):
    data = {
        "document": {
            "type": "PLAIN_TEXT",
            "content": text
        }
    }
    json_data = json.dumps(data)
    return json_data

def save_to_jsonl(jsonl_file, data):
    with open(jsonl_file, 'a') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

def get_existing_jsonl_length(jsonl_file):
    if not os.path.exists(jsonl_file):
        return 0
    with open(jsonl_file, 'r') as f:
        return sum(1 for _ in f)

def main(args):
    api_endpoint = "https://language.googleapis.com/v1/documents:moderateText"
    url_with_key = f"{api_endpoint}?key={args.api_key}"
    headers = {
        "Content-Type": "application/json"
    }

    # 读取 CSV 文件
    # jailbreak = pd.read_csv(args.input_csv_path, on_bad_lines='skip')
    jailbreak = pd.read_csv(args.input_csv_path, engine='python', encoding='utf-8')
    prompts = jailbreak[args.csv_col].tolist()

    # 读取已经存在的 JSONL 文件的行数
    existing_length = get_existing_jsonl_length(args.output_jsonl_path)
    print(f"Existing JSONL entries: {existing_length}")

    # 设置起始位置
    start_index = existing_length
    i = start_index

    # 设置要处理的最大数量
    max_count = args.max_prompts

    # 计算终止位置
    end_index = min(start_index + max_count, len(prompts)) if max_count > 0 else len(prompts)

    # 检查是否已经处理完所有的 prompts
    if i >= end_index:
        print("All prompts have been processed.")
        return

    # 继续处理剩余的 prompts
    results = []
    while i < end_index:
        print(f"{i+1} / {end_index}")
        data = prepare_text(prompts[i])
        retries = 0
        max_retries = 3
        success = False
        while retries < max_retries and not success:
            try:
                response = requests.post(url_with_key, headers=headers, data=data, timeout=10)
                if response.status_code == 200:
                    result = response.json()
                    # 检查 'moderationCategories' 或其他可能的字段
                    moderation_categories = result.get("moderationCategories", [])
                    if not moderation_categories and 'categories' in result:
                        moderation_categories = result['categories']

                    # 检查响应是否为空
                    if not result:
                        print("Empty response received, retrying...")
                        retries += 1
                        time.sleep(2)  # 等待2秒后重试
                        continue

                    print(f'prompt:{i}')
                    print(moderation_categories)
                    # 将 'moderationCategories' 添加回 'result'
                    result["moderationCategories"] = moderation_categories
                    result[args.csv_col] = prompts[i]
                    result["id"] = i  # 添加 id 字段
                    results.append(result)
                    i += 1
                    success = True
                    time.sleep(1.28)
                else:
                    print(f"Error {response.status_code}: {response.text}")
                    time.sleep(12.8)
                    retries += 1
                    continue
            except requests.exceptions.Timeout:
                print("Request timed out, retrying...")
                retries += 1
                time.sleep(2)  # 等待2秒后重试
                continue
            except Exception as e:
                print(f"Exception occurred: {e}")
                time.sleep(18.2)
                retries += 1
                continue

        if not success:
            print(f"Failed to process prompt at index {i}, skipping...")
            i += 1  # 跳过当前的prompt，处理下一个

        # 每当结果数量达到指定的 batch_size 时，保存到文件并清空 results 列表
        if len(results) == args.batch_size:
            save_to_jsonl(args.output_jsonl_path, results)
            results = []

    # 处理剩余的结果（如果有）
    if results:
        save_to_jsonl(args.output_jsonl_path, results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    file_name = "advbench_moderation_100"
    input_csv_path = f"/data/home/Xinzhe/GuardBreaker_new/data/csv/{file_name}.csv"
    colmn = judge_prompt.judge_colmn(input_csv_path)
    print("colmn为:")
    print(colmn)
    # "/data/home/Xinzhe/GuardBreaker/guard_attack/data/csv/{file_name}.csv"
    parser.add_argument('--input_csv_path', type=str, default=input_csv_path )
    parser.add_argument('--csv_col', type=str, default=colmn)
    parser.add_argument('--api_key', type=str, default="")
    # "/data/home/Xinzhe/GuardBreaker/guard_attack/data/score/{file_name}_google.jsonl"
    parser.add_argument('--output_jsonl_path', type=str, default=f"/data/home/Xinzhe/GuardBreaker_new/data/score/guard_attack/{file_name}_google.jsonl")
    parser.add_argument('--batch_size', type=int, default=5, help='Number of results to save at once')
    parser.add_argument('--max_prompts', type=int, default=21600, help='Maximum number of prompts to process (0 for all)')
    args = parser.parse_args()

    main(args)


