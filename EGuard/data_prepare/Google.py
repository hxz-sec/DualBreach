import requests
import json
import pandas as pd
from tqdm import trange
import time
import argparse


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

def main(args):
    api_endpoint = "https://language.googleapis.com/v1/documents:moderateText"
    url_with_key = f"{api_endpoint}?key={args.api_key}"
    headers = {
        "Content-Type": "application/json"
    }

    jailbreak = pd.read_csv(args.input_csv_path)
    prompts = jailbreak[args.csv_col].tolist()

    results = []
    i = 0
    while i < len(prompts):
        print(f"{i+1} / {len(prompts)}")
        data = prepare_text(prompts[i])
        try:
            response = requests.post(url_with_key, headers=headers, data=data)
            if response.status_code == 200:
                result = response.json()
                moderation_categories = result.get("moderationCategories", [])
                result["prompt"] = prompts[i]
                # print(result)
                results.append(result)
                i += 1
                time.sleep(1.28)
            else:
                time.sleep(12.8)
                continue
        except:
            time.sleep(18.2)
            continue

    save_to_jsonl(args.output_jsonl_path, results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv_path', type=str, default="/data/home/Xinzhe/GAN/data/normal_contents_2k.csv")
    parser.add_argument('--csv_col', type=str, default="prompts")
    parser.add_argument('--api_key', type=str, default="")
    parser.add_argument('--output_jsonl_path', type=str, default="./data/normal_google.jsonl")
    args = parser.parse_args()

    main(args)