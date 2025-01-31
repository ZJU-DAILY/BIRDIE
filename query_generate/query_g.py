import json
from tqdm import tqdm
import time
import requests
import threading
import random
import os
import argparse
import tqdm
from tqdm import tqdm
import tiktoken
import math
import numpy as np
import tiktoken
# from vllm import LLM, SamplingParams
from openai import OpenAI

client = OpenAI(
api_key="0",
base_url="http://localhost:{}/v1".format(os.environ.get("API_PORT", 8000))
)

# os.environ['http_proxy'] = "http://10.82.210.84:7890"
# os.environ['https_proxy'] = "http://10.82.210.84:7890"

def json_to_markdown(json_data):
    data = json_data
    columns = [col['text'] for col in data['columns']]
    rows = [row['cells'] for row in data['rows']]
    markdown_table = '|'
    for col in columns:
        if col == '':
            col = ' '
        markdown_table += col + '|'
    markdown_table += '\n|' + '|'.join(['---'] * len(columns)) + '|'
    for row in rows:
        markdown_table += '\n|'
        for cell in row:
            markdown_table += cell['text'] + '|'
    return markdown_table

def call_gpt_multiple(instruction,  document_title, prompt_example, n = 10, table_prompt_info = [], model = 'llm', use_example = False, prompt_example_num = 1):
    sample_list = table_prompt_info[1]
    temperature = 1
    retries = 0
    def get_prompt(use_example, instruction, document_title, sample_list, random_sample = False):
        if use_example:
            pass
        else:
            sample_num = 5 if table_prompt_info[2] >= 5 else max(table_prompt_info[2], 0)
            if len(sample_list) >= sample_num:
                result_sample = random.sample(sample_list, sample_num)
                sample_list = [sample for sample in sample_list if sample not in result_sample]
            else:
                sample_list = table_prompt_info[1]
                result_sample = random.sample(sample_list, sample_num)
                sample_list = [sample for sample in sample_list if sample not in result_sample]
            table_prompt = "\n".join(table_prompt_info[0] + result_sample)
            prompt_base =  " tableCaption: " + document_title + "\ntable: "+ table_prompt
            prompt = instruction + "\n\n" + prompt_base 
        return prompt, sample_list
    def is_valid_format(answer):
        try:
            answer_dict = json.loads(answer)
            return isinstance(answer_dict, dict) and set(answer_dict.keys()) == {'question'}
        except (json.JSONDecodeError, TypeError):
            return False
    
    prompt, sample_list = get_prompt(use_example, instruction, document_title, sample_list)  
    response = client.chat.completions.create(
            model = model,
            messages=[{"role": "user", "content": prompt}],
            temperature = temperature,
            n = 20,
            top_p = 1
        )
    answers = [choice.message.content for choice in response.choices]
    # print(answers)
    unique_answers = [answer for answer in list(set(answers)) if is_valid_format(answer)] 
    while len(unique_answers) < n and retries < 20:
        prompt, sample_list = get_prompt(use_example, instruction, document_title, sample_list) 
        additional_response = client.chat.completions.create(
            model = model,
            messages=[{"role": "user", "content": prompt}],
            temperature = temperature,
            n = 20,
            top_p = 1
        )
        additional_answers = [choice.message.content for choice in additional_response.choices]
        additional_answers = [answer for answer in additional_answers if is_valid_format(answer)]
        unique_answers.extend(additional_answers)
        unique_answers = list(set(unique_answers))  
        retries += 1
    return random.sample(unique_answers, n) if len(unique_answers) > n else unique_answers

def get_origin_id(table_path):
    table_id = []
    with open(table_path, 'r') as table_f:
        for line in table_f:
            table_data_f = json.loads(line)
            table_id.append(table_data_f['tableId'])
    table_id_map = {table_id[i]:i for i in range(len(table_id))}
    return table_id_map

def get_semantic_id(table_path):
    table_id_semantic_map = {}
    with open(table_path, 'r') as file:
        for line in file:
            try:
                data = json.loads(line.strip())
                table_id_semantic_map[data['tableID']] = data['semantic_id']
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from file {e}")
    return table_id_semantic_map

def prompt_length_exceeds_limit(prompt, model="gpt-3.5-turbo", max_tokens=1024):
    if "|---|" in prompt and len(prompt.split('\n')) <= 2:
        return False, 0
    encoding = tiktoken.encoding_for_model(model)
    token_count = len(encoding.encode(prompt))
    return token_count > max_tokens, token_count

def get_limit_prompt(table_prompt, token_count, max_tokens = 2048): 
    split_prompt = table_prompt.split("\n")
    header = split_prompt[:2]
    rows = split_prompt[2:]
    excess_tokens = token_count - max_tokens
    row_tokens = prompt_length_exceeds_limit("\n".join(rows), max_tokens = max_tokens)[1]
    row_token_count = row_tokens / len(rows)  
    num_rows_to_remove = int(excess_tokens / row_token_count) + 1
    num_rows_to_remove = num_rows_to_remove if num_rows_to_remove <= len(rows) else len(rows)
    rows = random.sample(rows, len(rows) - num_rows_to_remove)
    table_prompt = "\n".join(header + rows)
    return header, split_prompt[2:], len(rows) - num_rows_to_remove
def create_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_tag', type = str, default = "inc_0")
    parser.add_argument('--dataset_name', type=str, default='fetaqa')
    parser.add_argument('--tableid_path', type=str, default='/data1/zhh/baselines/GR4Table/tableidGenerate/hierarchical_clustering/docid/fetaqa/id_map.json')
    parser.add_argument('--num', type=int, default=20)
    parser.add_argument('--dataset_info', type=str, default='data/data_info.json')
    parser.add_argument('--out_train_path', type=str, default= 'table2question/prompt_test')
    parser.add_argument('--table_max_token', type = int, default = 2048)
    parser.add_argument("--model_name", default = "model/sentence-t5-base", type=str)
    parser.add_argument("--batch_size", default = 128, type=int)
    return parser.parse_args()

if __name__ == "__main__":
    args = create_args()
    os.makedirs(args.out_train_path, exist_ok=True)
    with open(args.dataset_info, 'r') as file:
        data_info = json.load(file)
    dataset = data_info[args.dataset_name]
    instructions = ['''Please generate 10 the most probable questions a user would ask based on this table, the question should be concise (no more than 15 words ) and the answers can be found in this table. The question should contain the explicit information of this table (such as the information of title) so that given the question and many tables, this table can be successfully found. Return in JSON format as {"question1":"", "question2":"", ......, "question10":""}.''',
                    '''Design one direct question for this table, ensuring that the question's answer can be inferred from the data in the table. The question should follow the format of the given examples (e.g., "where does the brazos river start and stop?", "who is the oldest wrestler that is still wrestling?",.......). The generated question should be based on the specific data provided in the table.\n\n Return the final result in JSON format as {"question":""}.''',
                    '''Please generate the probable question a user would ask based on this table, ensuring the answer can be found in some cells in this table or via an aggregation operator such as (Max,Min,Avg,Count). The generated question should contain the explicit information of this table (such as the information of the caption) so that given the question and a repository of tables, this table can be successfully found. \n\nReturn the final result in JSON format as {"question":""}.''',
                    '''Please generate the probable question a user would ask based on this table, ensuring the answer can be found in some cells in this table or via an aggregation operator such as (Max,Min,Avg,Count). The generated question should contain the explicit information of this table so that given the question and a repository of tables, this table can be successfully found. \n\nReturn the final result in JSON format as {"question":""}.''',
                    '''Please generate the probable question a user would ask based on this table, ensuring the answer can be found in some cells in this table or via an aggregation operator such as (𝑀𝑎𝑥,𝑀𝑖𝑛,𝐴𝑣𝑔,𝐶𝑜𝑢𝑛𝑡). The generated question should contain the explicit information of this table (such as the information of the caption) so that given the question and a repository of tables, this table can be successfully found. \n\n Return the final result in JSON format as {\"question\":\"\"}.''']
    instruction = instructions[-1]
    origin_id_map = get_origin_id(dataset['tables'])
    semantic_id_map = get_semantic_id(args.tableid_path)
    args.out_train_path = os.path.join(args.out_train_path, args.dataset_name)
    train_data_out_path = os.path.join(args.out_train_path, f"train_query_{args.run_tag}.json")
    with open(dataset['tables']) as table_f, open(train_data_out_path, 'w') as final_train_f:
        for line_num, line in enumerate(table_f):
            table_data_f = json.loads(line)
            try_count = 0
            response = None
            table_prompt = json_to_markdown(table_data_f)
            header_prompt = table_prompt.split("\n")[:2]
            example = []
            while response == None and try_count < 3:
                try_count += 1
                try:
                    max_token = args.table_max_token
                    exceeds_limit, token_count = prompt_length_exceeds_limit(table_prompt, max_tokens = max_token)
                    if exceeds_limit:
                        header_prompt, rows, row_num = get_limit_prompt(table_prompt, token_count, max_tokens = max_token)
                    else:
                        rows = table_prompt.split("\n")[2:]
                        row_num = len(rows)
            
                    response = call_gpt_multiple(instruction, table_data_f["documentTitle"],  example, table_prompt_info=(header_prompt, rows, row_num), n= args.num)
                except Exception as e:
                    print(e)
                    continue
            doc_id = origin_id_map[table_data_f['tableId']]
            sem_id = semantic_id_map[table_data_f['tableId']]
            try:
                print(line_num)
                for question in response:
                    wait2write = {"text_id": sem_id,
                                    "question": f"{question}",
                                    "tableId" : table_data_f['tableId'],
                                    "origin_id": doc_id,
                                    }
                    final_train_f.write(json.dumps(wait2write) + '\n')
                    print(wait2write)
            except Exception as e:
                print(e)