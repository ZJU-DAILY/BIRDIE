import argparse
import warnings
from collections import Counter
import itertools
from sentence_transformers import SentenceTransformer, util
from data import IndexingTrainDataset, IndexingCollator
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from inspect import signature
from transformers import (
    MT5Tokenizer,
    MT5TokenizerFast,
    MT5ForConditionalGeneration,
    TrainingArguments,
    TrainerCallback,
    MT5Tokenizer,
    MT5TokenizerFast,
    MT5ForConditionalGeneration,
    HfArgumentParser,
    IntervalStrategy,
    set_seed,
)
from trainer import DSITrainer, MapTrainer
import numpy as np
import torch
import wandb
import os
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import Optional
import json
from tqdm import tqdm
# wandb.init(mode='offline')
set_seed(313)

@dataclass
class RunArguments:
    model_name: str = field(default="google/mt5-base")
    base_model_path: str = field(default="./model/mt5-base")
    max_length: Optional[int] = field(default=64)
    id_max_length: Optional[int] = field(default=20)
    train_file: str = field(default="./dataset/fetaqa_inc/train_0.json")
    valid_file: str = field(default="./dataset/fetaqa_inc/test_0.json")
    all_file: str = field(default="./dataset/fetaqa_inc/train_0.json", metadata={"help": "the valid tabids"})
    task: str = field(default="Index",  metadata={"help": "Index, Search"})
    top_k: Optional[int] = field(default=10)
    num_return_sequences: Optional[int] = field(default=10)
    q_max_length: Optional[int] = field(default=32)
    remove_prompt: Optional[bool] = field(default=True)
    peft: Optional[bool] = field(default=False)


@dataclass
class CustomTrainingArguments(TrainingArguments):
    num: int = field(default=0)
    LoRA_1: str = field(default="None")
    LoRA_2: str = field(default="None")
    LoRA_3: str = field(default="None")
    partition_0: str = field(default="None")
    partition_1: str = field(default="None")
    partition_2: str = field(default="None")
    partition_3: str = field(default="None")
    num_train_epochs: int = field(default=3)
    learning_rate: float = field(default=0.0005)
    warmup_steps: float = field(default=10000)
    per_device_train_batch_size: int = field(default=64)
    per_device_eval_batch_size: int = field(default=32)
    evaluation_strategy: str = field(default="steps")
    eval_steps: int = field(default=100)
    max_steps: int = field(default=800)
    save_strategy: str = field(default="steps")
    dataloader_num_workers: int = field(default=6)
    save_steps: int = field(default=100)
    save_total_limit: int = field(default=3)
    gradient_accumulation_steps : int = field(default=6)
    report_to: str = field(default="wandb")
    logging_steps: int = field(default=100)
    metric_for_best_model: str = field(default="Hits@5")
    dataloader_drop_last: bool = field(default=False)
    greater_is_better: bool = field(default=True)
    run_name: str = field(default="feta_inc")
    output_dir: str = field(default="./model/feta_inc0")

class PrintGradientsCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs['model']
        print(f"Epoch {state.epoch} ended.")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"Parameter Name: {name}")
                print("Parameter Value:")
                print(param.data)
                if param.grad is not None:
                    print("Parameter Gradient:")
                    print(param.grad)
                else:
                    print("Parameter Gradient: None")

def find_similar_embeddings(embeddings_1, embeddings_2, topk =1):
    similarity_matrix = np.dot(embeddings_1, embeddings_2.T)
    topk_indices = np.argsort(similarity_matrix, axis=1)[:, -topk:]
    # Sort the top-k indices for each row in ascending order
    topk_indices = np.sort(topk_indices)
    # max_indices = np.argmax(similarity_matrix, axis=1)
    return topk_indices

def inference_on_test_set(model, test_args, test_dataset, tokenizer, restrict_decode_vocab, id_max_length):
    trainer = MapTrainer(
        model=model,
        args=test_args,
        tokenizer=tokenizer,
        data_collator=IndexingCollator(tokenizer=tokenizer),
        restrict_decode_vocab=restrict_decode_vocab,
        id_max_length=id_max_length
    )

    # 执行推理
    results = trainer.predict(test_dataset=test_dataset)

    predictions = results.predictions
    labels = results.label_ids

    return predictions, labels

def make_compute_metrics(tokenizer, valid_ids):

    def compute_metrics(eval_preds):
        hit_at_1 = 0
        hit_at_5 = 0
        i = 0
        gt2err = dict()
        for beams, label in zip(eval_preds.predictions, eval_preds.label_ids):
            rank_list = tokenizer.batch_decode(beams, skip_special_tokens=True)

            label_id = tokenizer.decode(label, skip_special_tokens=True)
            # filter out duplicates and invalid docids
            filtered_rank_list = []
            for docid in rank_list:
                if docid not in filtered_rank_list and docid in valid_ids:
                    filtered_rank_list.append(docid)
            hits = np.where(np.array(filtered_rank_list)[:5] == label_id)[0]
            # hits = np.where(np.array(filtered_rank_list)[:10] == label_id)[0]

            if len(hits) != 0:
                hit_at_5 += 1
                if hits[0] == 0:
                    hit_at_1 += 1
            i += 1
        return {"Hits@1": hit_at_1 / len(eval_preds.predictions), "Hits@5": hit_at_5 / len(eval_preds.predictions)}
    return compute_metrics

def valid_id(train_pth):
    valid_id_set = set()
    with open(train_pth, "r") as f:
        for line in f:
            data = json.loads(line)
            valid_id_set.add(str(data['text_id']))
    return valid_id_set

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    parser = HfArgumentParser((CustomTrainingArguments, RunArguments))
    args, run_args = parser.parse_args_into_dataclasses()

    # We use wandb logger: https://wandb.ai/site.
    if args.local_rank == 0:  # only on main process
        # Initialize wandb run
        wandb.login()
        wandb.init(project="DSI2Table", name=args.run_name, entity="DSITable")

    tokenizer = MT5Tokenizer.from_pretrained(run_args.base_model_path, cache_dir='cache')
    fast_tokenizer = MT5TokenizerFast.from_pretrained(run_args.base_model_path, cache_dir='cache')

    # legal tokens
    SPIECE_UNDERLINE = "▁"
    INT_TOKEN_IDS = []
    for token, id in tokenizer.get_vocab().items():
        if token[0] == SPIECE_UNDERLINE:
            if token[1:].isdigit():
                INT_TOKEN_IDS.append(id)
        if token == SPIECE_UNDERLINE:
            INT_TOKEN_IDS.append(id)
        elif token.isdigit():
            INT_TOKEN_IDS.append(id)
    INT_TOKEN_IDS.append(tokenizer.eos_token_id)

    def restrict_decode_vocab(batch_idx, prefix_beam):
        return INT_TOKEN_IDS

    if run_args.task == "Index":
        training_args = args
        model = MT5ForConditionalGeneration.from_pretrained(run_args.base_model_path, cache_dir='cache')
        if run_args.peft:
            print("**************PEFT*****************")
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                inference_mode=False,
                target_modules='(decoder\.block\.\d+\.layer\.2|encoder\.block\.\d+\.layer\.1)\.DenseReluDense\.(wi_0|wi_1|wo)',
                r=8,
                lora_alpha=32,
                lora_dropout=0.1
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        else:
            print("**************FULL*****************")

        train_dataset = IndexingTrainDataset(path_to_data=run_args.train_file,
                                             max_length=run_args.max_length,
                                             cache_dir='cache',
                                             tokenizer=tokenizer)

        valid_dataset = IndexingTrainDataset(path_to_data=run_args.valid_file,
                                             max_length=run_args.max_length,
                                             cache_dir='cache',
                                             remove_prompt=run_args.remove_prompt,
                                             tokenizer=tokenizer)
        all_dataset = IndexingTrainDataset(
            path_to_data=run_args.all_file,
            max_length=run_args.max_length,
            cache_dir='cache',
            tokenizer=tokenizer)

        trainer = DSITrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=IndexingCollator(
                tokenizer,
                padding='longest',
            ),
            compute_metrics = make_compute_metrics(fast_tokenizer, all_dataset.valid_ids),
            restrict_decode_vocab = restrict_decode_vocab,
            id_max_length=run_args.id_max_length,
        )
        trainer.train()


    elif run_args.task == 'Search':
        test_args = args
        test_args.do_train = False
        test_args.do_predict = True
        test_dataset = IndexingTrainDataset(path_to_data=run_args.valid_file,
                                            max_length=run_args.max_length,
                                            cache_dir='cache',
                                            remove_prompt=run_args.remove_prompt,
                                            tokenizer=tokenizer)
        test_q = []
        for data in test_dataset.train_data:
            test_q.append(str(data['text']))

        test_num = test_dataset.total_len
        print("test_num:{}".format(test_num))

        base_model = MT5ForConditionalGeneration.from_pretrained(run_args.base_model_path, cache_dir='cache')

        Lora_pths = [None, args.LoRA_1, args.LoRA_2, args.LoRA_3]
        partition_pths = [args.partition_0, args.partition_1, args.partition_2]
        Lora_pths = Lora_pths[:args.num]
        partition_pths = partition_pths[:args.num]
        cadidates_ids = [[] for _ in range(test_num)]
        '''
          candidates_ids = [ [ [model_0], [model_1]...], ...]
        '''
        lable_ids = []

        id2q = [dict() for _ in range(len(partition_pths))]
        '''
          id2q = [ {"text_id1": [q1, q2, ...]}, {"text_id2": [q3, q4, ...]} ]
        '''
        for i, (valid_pth, lora_pth) in enumerate(zip(partition_pths, Lora_pths)):
            valid_ids = valid_id(valid_pth)
            with open(valid_pth, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    k = data["text_id"]
                    q = data["text"]
                    if k not in id2q[i].keys():
                        id2q[i][k] = []  # 第一个value是table
                        # id2q[i][k].append(q)
                    else:
                        id2q[i][k].append(q)

            if lora_pth is not None:
                model = PeftModel.from_pretrained(base_model, lora_pth)
            else:
                model = base_model
            predictions, labels = inference_on_test_set(
                model=model,
                test_args=test_args,
                test_dataset=test_dataset,
                tokenizer=tokenizer,
                restrict_decode_vocab=restrict_decode_vocab,
                id_max_length=run_args.id_max_length
            )
            j = 0
            for beams, label in zip(predictions, labels):
                rank_list = tokenizer.batch_decode(beams, skip_special_tokens=True)
                label_id = tokenizer.decode(label, skip_special_tokens=True)
                lable_ids.append(label_id)
                filtered_ids = []
                for docid in rank_list:
                    if docid not in filtered_ids and docid in valid_ids:
                        filtered_ids.append(docid)
                cadidates_ids[j].append(filtered_ids[:5])
                j += 1
        hit_at_5 = 0
        hit_at_1 = 0
        model = SentenceTransformer("./model/stsb-roberta-base")
        merged_id2q = {}

        for d in id2q:
            merged_id2q.update(d)

        for i, q in enumerate(test_q):
            label_id = lable_ids[i]
            ids = list(itertools.chain(*cadidates_ids[i]))
            gen_qs = [merged_id2q[id] for id in ids]
            gen_qs = list(itertools.chain(*gen_qs))
            # for model_k_candidates in cadidates_ids[i]:
            #     gen_qs = [merged_id2q[id] for id in model_k_candidates]
            #     gen_qs = list(itertools.chain(*gen_qs))
            #     lens.append(len(gen_qs))
            embed = model.encode(gen_qs, batch_size=2048)
            candidate_embeddings = np.array(embed)
            candidate_embeddings = candidate_embeddings / np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
            test_q_embeddings = model.encode(q)
            test_q_embeddings = test_q_embeddings.reshape(1, -1)
            test_q_embeddings = test_q_embeddings / np.linalg.norm(test_q_embeddings, axis=1, keepdims=True)
            max_ids = find_similar_embeddings(test_q_embeddings, candidate_embeddings, topk=5)
            t = v = 0
            id2model = dict()
            for k, sub_lst in enumerate(cadidates_ids[i]):
                for v in range(t, t + len(sub_lst) * 20):
                    id2model[v] = k
                t = v + 1
            maxids = max_ids.flatten().tolist()
            model_nums = [id2model[id] for id in maxids]
            count = Counter(model_nums)
            model_num = count.most_common(1)[0][0]
            preds = cadidates_ids[i][model_num]
            hits = np.where(np.array(preds)[:5] == label_id)[0]
            if len(hits) != 0:
                hit_at_5 += 1
                if hits[0] == 0:
                    hit_at_1 += 1
        print("Hits@1: {}".format(hit_at_1 / test_num))
        print("Hits@5: {}".format(hit_at_5 / test_num))

    else:
        raise NotImplementedError("--task should be in 'DSI' or 'inference'")


if __name__ == "__main__":

    main()






