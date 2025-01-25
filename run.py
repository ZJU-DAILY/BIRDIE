import argparse
import warnings
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
from trainer import DSITrainer
import numpy as np
import torch
import wandb
import os
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import Optional
import json
from tqdm import tqdm
wandb.init(mode='offline')


@dataclass
class RunArguments:
    model_name: str = field(default="google/mt5-base")
    base_model_path: str = field(default="./model/mt5-base")
    max_length: Optional[int] = field(default=64)
    id_max_length: Optional[int] = field(default=20)
    train_file: str = field(default="./dataset/fetaqa/train.json")
    valid_file: str = field(default="./dataset/fetaqa/test.json")
    task: str = field(default="Index",  metadata={"help": "Index, Search"})
    top_k: Optional[int] = field(default=10)
    num_return_sequences: Optional[int] = field(default=10)
    q_max_length: Optional[int] = field(default=32)
    remove_prompt: Optional[bool] = field(default=True)
    peft: Optional[bool] = field(default=False)

@dataclass
class CustomTrainingArguments(TrainingArguments):
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
    run_name: str = field(default="feta")
    output_dir: str = field(default="./model/feta")


set_seed(313)
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
            compute_metrics = make_compute_metrics(fast_tokenizer, train_dataset.valid_ids),
            restrict_decode_vocab = restrict_decode_vocab,
            id_max_length=run_args.id_max_length,
        )
        trainer.train()


    elif run_args.task == 'Search':
        test_args = args
        test_args.do_train = False
        test_args.do_predict = True
        if run_args.peft == True:
            model = MT5ForConditionalGeneration.from_pretrained(run_args.base_model_path, cache_dir='cache')
            lora_model_path = run_args.lora_model_path
            model = PeftModel.from_pretrained(model, lora_model_path)
        else:
            model = MT5ForConditionalGeneration.from_pretrained(run_args.base_model_path, cache_dir='cache')
        test_dataset = IndexingTrainDataset(path_to_data=run_args.valid_file,
                                             max_length=run_args.max_length,
                                             cache_dir='cache',
                                             remove_prompt=run_args.remove_prompt,
                                             tokenizer=tokenizer)

        all_dataset = IndexingTrainDataset(path_to_data=
                                           run_args.train_file,
                                           max_length=run_args.max_length,
                                           cache_dir='cache',
                                           remove_prompt=run_args.remove_prompt,
                                           tokenizer=tokenizer)
        # init trainer
        trainer = DSITrainer(
            model=model,
            tokenizer=tokenizer,
            args=test_args,
            compute_metrics=make_compute_metrics(fast_tokenizer, all_dataset.valid_ids),
            data_collator=IndexingCollator(
                tokenizer,
                padding='longest',
            ),
            restrict_decode_vocab=restrict_decode_vocab,
            id_max_length=run_args.id_max_length)

        trainer.evaluate(test_dataset)

    else:

        raise NotImplementedError("--task should be in 'DSI' or 'inference'")


if __name__ == "__main__":
    main()
