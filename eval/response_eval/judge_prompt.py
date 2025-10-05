import csv
import json
import os


def judge_colmn(file_path):
    # 获取文件扩展名
    _, file_extension = os.path.splitext(file_path)
    
    # 根据扩展名判断文件类型
    if file_extension.lower() == '.jsonl':
        return check_jsonl_fields(file_path)
    elif file_extension.lower() == '.csv':
        return check_csv_fields(file_path)
    else:
        return "文件不是csv或jonsl类型"

def check_csv_fields(file_path):
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # 读取第一行（字段名）
        #required_adv = 'prompt_with_adv'
        #required_prompt= 'prompt'
        #required_harmfful = 'harmful'
        #flag_adv = required_adv.issubset(headers)
        #flag_prompt = required_prompt.issubset(headers)
        #flag_harmful = required_harmfful.issubset(headers)
        if 'prompt_with_adv' in headers:
            return "prompt_with_adv"
        else:
            if 'prompt' in headers:
                return "prompt"
            else :
                if 'harmful' in headers :
                    return "harmful"
                else :
                    return "没有符合的colmn"                     # 检查是否包含所有必需的字段


def check_jsonl_fields(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        first_line = file.readline().strip()  # 读取第一行并去掉空白字符
        first_line_data = json.loads(first_line)

        if first_line_data.get("prompt_with_adv", ""):
            prompt = "prompt_with_adv"
        else:    
            if first_line_data.get("prompt", ""):
                prompt = "prompt"
            else :
                if first_line_data.get("harmful", ""):
                    prompt = "harmful"
                else :
                    return "没有符合的colmn"
    count=0
    count_adv=0
    count_prompt=0
    count_harmful=0
    with open(file_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            count=count+1
            line_data = json.loads(line)
            if line_data.get("prompt_with_adv", ""):
                count_adv=count_adv+1
            else:    
                if first_line_data.get("prompt", ""):
                    count_prompt=count_prompt+1
                else :
                    if first_line_data.get("harmful", ""):
                        count_harmful=count_harmful+1
                    else :
                        return "没有符合的colmn"
    print(count_adv)
    print(count_prompt)
    print(count_harmful)
    if count_adv != count and count_prompt != count and count_harmful != count :
        return "数据字段不统一"
    else :
        return prompt       


# 运行主函数
if __name__ == '__main__':
    #file_name = "advbench_role_100"
    
    input_jsonl_file = f"/data/home/Xinzhe/GuardBreaker/guard_attack/data/safe_label/harmBench_role_5500_safe_guard3.jsonl"
    colmn = judge_colmn(input_jsonl_file)

    print(input_jsonl_file)
    print("colmn为：")
    print(colmn)

