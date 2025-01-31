from sentence_transformers import SentenceTransformer, util
import os
import numpy as np
import json
import random
import argparse
import os
import pandas as pd
from tqdm import tqdm

def table2text(table_data):
    columns = [column["text"] for column in table_data["columns"]]

    table_string = ""
    for row in table_data["rows"]:
        row_string = ""
        for i, cell in enumerate(row["cells"]):
            row_string += f"{columns[i]}: {cell['text']}, "
        row_string = row_string[:-2] + "\n"
        table_string += row_string

    return table_string

def remove_duplicates(lst):
    unique_list = []
    for item in lst:
        if item not in unique_list:
            unique_list.append(item)
    return unique_list


def encode_sentences_to_vectors(sentences, model, batch_size=32):
    embeddings = []
    for i in tqdm(
        range(0, len(sentences), batch_size), desc="Processing Sentences", unit="batch"
    ):
        batch_sentences = sentences[i : i + batch_size]
        batch_embeddings = model.encode(batch_sentences)
        embeddings.extend(batch_embeddings)

    embeddings_np = np.array(embeddings)

    return embeddings_np


def add_suffix_to_duplicates(string_list):
    seen = {}
    result = []
    for string in string_list:
        if string in seen:
            seen[string] += 1
            new_string = string + str(seen[string])
            result.append(new_string)
        else:
            seen[string] = 0
            result.append(string)
    return result


def get_table_data(data_info, key):
    tables_path = data_info[key]["tables"]
    with open(tables_path, "r") as table_f:
        tableId_map, table_data_list, tabel_title_schema = [], [], []
        for linenum, line in enumerate(table_f):
            table_data_tf = json.loads(line)

            table_title = table_data_tf["documentTitle"]
            table_schema = ",".join(
                sorted(
                    [column["text"] for column in table_data_tf["columns"]],
                    key=lambda x: x[0] if len(x) > 0 else "",
                )
            )
            table_data = table2text(table_data_tf)
            tabel_title_schema.append(table_title + "\n" + table_schema)
            table_data_list.append(table_data)
            tableId_map.append((linenum, table_data_tf["tableId"]))
        return tabel_title_schema, table_data_list, tableId_map


def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="BIRDIE/tableid/temp", type=str)
    parser.add_argument("--dataset_name", default="fetaqa_0", type=str)
    parser.add_argument(
        "--dataset_info", default="BIRDIE/tableid/data_info.json", type=str
    )
    parser.add_argument("--model_name", default="model/sentence-t5-base", type=str)
    parser.add_argument("--batch_size", default=256, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    # random.seed(313)
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    args = create_args()

    run_tag = args.dataset_name

    model = SentenceTransformer(args.model_name)
    with open(args.dataset_info, "r") as file:
        data_info = json.load(file)
    table_title_schema, table_data_list, tableId_map = get_table_data(
        data_info, args.dataset_name
    )

    table_title_schema_embedding = encode_sentences_to_vectors(
        table_title_schema, model, args.batch_size
    )
    table_data_embedding = encode_sentences_to_vectors(
        table_data_list, model, args.batch_size
    )
    save_path = os.path.join(args.save_dir, args.dataset_name)
    os.makedirs(save_path, exist_ok=True)
    np.save(
        os.path.join(save_path, f"table_title_schema_embedding_{run_tag}.npy"),
        table_title_schema_embedding,
    )
    np.save(
        os.path.join(save_path, f"table_data_embedding_{run_tag}.npy"),
        table_data_embedding,
    )
    df = pd.DataFrame(tableId_map, columns=["doc_id", "table_id"])
    df.to_csv(os.path.join(save_path, f"tableId_list_{run_tag}.csv"), index=False)

    print("done")
