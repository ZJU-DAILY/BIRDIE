import time
from typing import Dict, List, Tuple, Optional, Any, Union
from transformers.trainer import Trainer, PredictionOutput
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.distributed as dist
from sklearn.cluster import KMeans

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class DSITrainer(Trainer):
    def __init__(self, restrict_decode_vocab, id_max_length, **kwds):
        super().__init__(**kwds)
        self.restrict_decode_vocab = restrict_decode_vocab
        self.id_max_length = id_max_length
        self.infer_time = 0

    def compute_loss(self, model, inputs, return_outputs=False):
        loss = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['labels']).loss
        if return_outputs:
            return loss, [None, None]  # fake outputs
        return loss


    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        model.eval()
        # eval_loss = super().prediction_step(model, inputs, True, ignore_keys)[0]
        inputs['labels'] = inputs['labels'].to(self.args.device)
        with torch.no_grad():
            # Greedy search
            # doc_ids = model.generate(
            #     inputs['input_ids'].to(self.args.device),
            #     max_length=20,
            #     prefix_allowed_tokens_fn=self.restrict_decode_vocab,
            #     early_stopping=True,)

            # Beam search
            # print(model.generate.__doc__)
            start_time = time.time()
            batch_beams = model.generate(
                input_ids=inputs['input_ids'].to(self.args.device),
                max_length=20,
                num_beams=20,
                prefix_allowed_tokens_fn=self.restrict_decode_vocab,
                num_return_sequences=20,
                early_stopping=True, )
            end_time = time.time()
            inftime = end_time-start_time
            self.infer_time += inftime

            if batch_beams.shape[-1] < self.id_max_length:
                batch_beams = self._pad_tensors_to_max_len(batch_beams, self.id_max_length)

            inputs['labels'] = self._pad_tensors_to_max_len(inputs['labels'], self.id_max_length)

            batch_beams = batch_beams.reshape(inputs['input_ids'].shape[0], 20, -1)

        return (None, batch_beams, inputs['labels'])

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")
        tensor[tensor == -100] = self.tokenizer.pad_token_id
        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor




class LossCalculationTrainer(Trainer):
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        # Get the logits and the labels
        labels = inputs.get("labels")
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss
        return loss, outputs.logits, labels



class MultiTrainer(Trainer):
    def __init__(self, restrict_decode_vocab, id_max_length, **kwds):
        super().__init__(**kwds)
        self.restrict_decode_vocab = restrict_decode_vocab
        self.id_max_length = id_max_length

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        model.eval()
        inputs['labels'] = inputs['labels'].to(self.args.device)

        with torch.no_grad():
            # Beam search with output scores
            outputs = model.generate(
                input_ids=inputs['input_ids'].to(self.args.device),
                max_length=20,
                num_beams=20,
                prefix_allowed_tokens_fn=self.restrict_decode_vocab,
                num_return_sequences=20,
                early_stopping=True,
                output_scores=True,  # Output scores for confidence calculation
                return_dict_in_generate=True
            )

            batch_beams = outputs.sequences
            sequences_scores = outputs.sequences_scores

            # Pad sequences if necessary
            if batch_beams.shape[-1] < self.id_max_length:
                batch_beams = self._pad_tensors_to_max_len(batch_beams, self.id_max_length)

            inputs['labels'] = self._pad_tensors_to_max_len(inputs['labels'], self.id_max_length)
            batch_beams = batch_beams.reshape(inputs['input_ids'].shape[0], 20, -1)

            # Convert sequence scores to probabilities (optional normalization step)
            confidences = torch.exp(sequences_scores)  # Convert log-probabilities to probabilities

        return (confidences, batch_beams, inputs['labels'])


    def predict(self,
                test_dataset: Dataset, ignore_keys: Optional[List[str]] = None,
                metric_key_prefix: str = "test"
                )-> PredictionOutput:


        self.model.eval()

        all_confidences = []
        all_predictions = []
        all_labels = []

        for inputs in self.get_test_dataloader(test_dataset):

            confidences, batch_beams, labels = self.prediction_step(
                self.model, inputs, prediction_loss_only=False, ignore_keys=ignore_keys
            )


            all_confidences.append(confidences)
            all_predictions.append(batch_beams)
            all_labels.append(labels)


        all_confidences = torch.cat(all_confidences, dim=0)
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)


        return PredictionOutput(predictions=all_predictions, label_ids=all_labels, metrics={"confidences": all_confidences})

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")
        tensor[tensor == -100] = self.tokenizer.pad_token_id
        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor


    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """
        Returns the test [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            test_dataset (`torch.utils.data.Dataset`, *optional*):
                The test dataset to use. If it is a [`~datasets.Dataset`], columns not accepted by the
                `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        data_collator = self.data_collator

        # if is_datasets_available() and isinstance(test_dataset, datasets.Dataset):
        #     test_dataset = self._remove_unused_columns(test_dataset, description="test")
        # else:
        #     data_collator = self._get_collator_with_removed_columns(data_collator, description="test")
        #
        # if isinstance(test_dataset, torch.utils.data.IterableDataset):
        #     if self.args.world_size > 1:
        #         test_dataset = IterableDatasetShard(
        #             test_dataset,
        #             batch_size=self.args.eval_batch_size,
        #             drop_last=self.args.dataloader_drop_last,
        #             num_processes=self.args.world_size,
        #             process_index=self.args.process_index,
        #         )
        #     return DataLoader(
        #         test_dataset,
        #         batch_size=self.args.eval_batch_size,
        #         collate_fn=data_collator,
        #         num_workers=self.args.dataloader_num_workers,
        #         pin_memory=self.args.dataloader_pin_memory,
        #     )

        test_sampler = self._get_eval_sampler(test_dataset)

        # We use the same batch_size as for eval.
        return DataLoader(
            test_dataset,
            sampler=test_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            shuffle=False,
        )



class MapTrainer(Trainer):
    def __init__(self, restrict_decode_vocab, id_max_length, **kwds):
        super().__init__(**kwds)
        self.restrict_decode_vocab = restrict_decode_vocab
        self.id_max_length = id_max_length

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        model.eval()


        inputs['labels'] = inputs['labels'].to(self.args.device)
        with torch.no_grad():
            batch_beams = model.generate(
                input_ids=inputs['input_ids'].to(self.args.device),
                max_length=20,
                num_beams=20,
                prefix_allowed_tokens_fn=self.restrict_decode_vocab,
                num_return_sequences=20,
                early_stopping=True, )

            if batch_beams.shape[-1] < self.id_max_length:
                batch_beams = self._pad_tensors_to_max_len(batch_beams, self.id_max_length)

            inputs['labels'] = self._pad_tensors_to_max_len(inputs['labels'], self.id_max_length)

            batch_beams = batch_beams.reshape(inputs['input_ids'].shape[0], 20, -1)

        return (None, batch_beams, inputs['labels'])


    def predict(self,
                test_dataset: Dataset, ignore_keys: Optional[List[str]] = None,
                metric_key_prefix: str = "test"
                )-> PredictionOutput:

        # 设置模型为评估模式
        self.model.eval()

        # 定义用于存储的变量
        all_confidences = []
        all_predictions = []
        all_labels = []

        for inputs in self.get_test_dataloader(test_dataset): #no shuffle保证test样本顺序正确
            # 获取单个批次的预测结果
            _, batch_beams, labels = self.prediction_step(
                self.model, inputs, prediction_loss_only=False, ignore_keys=ignore_keys
            )
            all_predictions.append(batch_beams)
            all_labels.append(labels)

        # 将所有批次的结果合并

        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        metrics = {}

        # 返回结果
        return PredictionOutput(predictions=all_predictions, label_ids=all_labels, metrics=metrics)

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")
        tensor[tensor == -100] = self.tokenizer.pad_token_id
        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor


    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """
        Returns the test [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            test_dataset (`torch.utils.data.Dataset`, *optional*):
                The test dataset to use. If it is a [`~datasets.Dataset`], columns not accepted by the
                `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        data_collator = self.data_collator


        test_sampler = self._get_eval_sampler(test_dataset)

        # We use the same batch_size as for eval.
        return DataLoader(
            test_dataset,
            sampler=test_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            shuffle=False,
        )

