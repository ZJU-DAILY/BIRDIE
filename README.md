# **BIRDIE: Natural Language-Driven Table Discovery Using Differentiable Search Index**

BIRDIE: An effective NL-driven table discovery framework using a differentiable search index. BIRDIE first assigns each table a prefix-aware identifier and leverages a LLM-based query generator to create synthetic queries for each table.  It then encodes the mapping between synthetic queries/tables and their corresponding table identifiers into the parameters of an encoder-decoder language model, enabling deep query-table interactions. During search, the trained model directly generates table identifiers for a given query. To accommodate the continual indexing of dynamic tables, we introduce an index update strategy via parameter isolation, which mitigates the issue of catastrophic forgetting.

## Requirements

* Python 3.7
* PyTorch 1.10.1
* CUDA 11.5
* NVIDIA 4090 GPUs

Please refer to the source code to install all required packages in Python.

## Datasets

We use three benchmark datasets NQ-Tables, FetaQA, and [OpenWikiTable](https://github.com/sean0042/Open_WikiTable). 
NQ-Tables and FetaQA can be downloaded from [here](https://github.com/TheDataStation/solo/blob/main/get_data.sh). OpenWikiTable should be pre-processed to be the same format with the other two datasets.

## Run Experimental Case

**Scenario I : Indexing from scratch**

+ Data preparation

  - Assign a table id for each table in the repository

    First, modify the dataset path in the data_info.json, and generate the representations of all tables

    ```
    cd BIRDIE/tableid/
    python emb.py --dataset_name "fetaqa" 
    ```

    Second, generate semantic IDs for each table through hierarchical clustering

    ```
    python hierarchical_clustering.py --dataset_name "fetaqa" --semantic_id_dir "BIRDIE/tableid/docid/"
    ```


  - Generate synthetic queries for each table

    Download the query generators and start the [vllm](https://github.com/vllm-project/vllm) service, then run the code below

    ```
    cd BIRDIE/query_generate/
    python query_g.py --dataset_name "fetaqa" --num 20 --tableid_path [Your path] --out_train_path [Your path]
    ```

    **tableid_path** is the path to the tableid file, **num** is the number of synthetic queries for each table, **out_train_path** is the path to  the output file.

+ Train the model to index the tables in the repository

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python3 -m torch.distributed.launch --nproc_per_node=6 run.py \
--task "Index" \
--train_file "./dataset/fetaqa/train.json" \ 
--valid_file "./dataset/fetaqa/test.json" \
--gradient_accumulation_steps 6 \
--max_steps 8000 \
--run_name "feta" \
--output_dir "./model/feta"
```

+ Search using the trained model

```
CUDA_VISIBLE_DEVICES=0 python3 run.py \
--task "Search" \
--train_file "./dataset/fetaqa/train.json" \
--valid_file "./dataset/fetaqa/test.json" \
--base_model_path "./model/feta/checkpoint-8000" \
--output_dir "./model/feta"
```

**Scenario II : Index Update**

+ Data preparation for D<sup>0</sup>

  - Assign a tabid for each table in the repository D<sup>0</sup>

    First, generate the representations of tables.

    ```
    cd BIRDIE/tableid/
    python emb.py --dataset_name "fetaqa_inc_0" 
    ```

    Second, generate semantic IDs for each table through hierarchical clustering.

    ```
    python hierarchical_clustering.py --dataset_name "fetaqa_inc_0" --semantic_id_dir "BIRDIE/tableid/docid/"
    ```

  - Generate synthetic queries for each table in D<sup>0</sup>, similar to the steps in "Indexing from scratch"

+ Train the model M<sup>0  </sup> on the repository D<sup>0</sup>

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python3 -m torch.distributed.launch --nproc_per_node=6 run_cont.py \
--task "Index" \
--train_file "./dataset/fetaqa_inc/train_0.json" \
--valid_file "./dataset/fetaqa_inc/test_0.json" \
--gradient_accumulation_steps 6 \
--max_steps 7000 --run_name "feta_inc0" \
--output_dir "./model/feta_inc0" 
```

+ Data preparation for D<sup>1</sup>

  - Assign a tabid for each table in the repository D<sup>1</sup> by running the incremental tabid assign algorithm

    First,  generate the representations of tables

    ```
    cd BIRDIE/tableid/
    python emb.py --dataset_name "fetaqa_inc_1" 
    ```

    Second, generate semantic IDs for each table through incremental tabid assign algorithm

    ```
    python cluster_tree.py --dataset_name "fetaqa_inc_1" --base_tag "fetaqa_inc_0"
    ```

  - Generate synthetic queries for each table in D<sup>1</sup>, similar to the steps in "Indexing from scratch"

+ Train a memory unit L<sup>1</sup> to index D<sup>1</sup> based on the model M<sup>0</sup> using LoRA

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python3 -m torch.distributed.launch --nproc_per_node=6 run_cont.py \
--task "Index" \
--base_model_path "./model/feta_inc0/checkpoint-7000" \
--train_file "./dataset/fetaqa_inc/train_1.json" \
--valid_file "./dataset/fetaqa_inc/test_1.json" \
--peft True \
--gradient_accumulation_steps 6 \
--max_steps 4000 --run_name "feta_LoRA_d1" \
--output_dir "./model/feta_LoRA_d1"
```

+ Search tables using the model M<sup>0 </sup> and the plug-and-play LoRA memory L<sup>1</sup>

```
CUDA_VISIBLE_DEVICES=0 python3 run_cont.py \
--task "Search" \
--valid_file \
"./dataset/fetaqa_inc/test_0+1.json" \
--LoRA_1 "./model/feta_LoRA_d1/checkpoint-4000" \
--num 2 \
--partition_0 "./dataset/fetaqa_inc/train_0.json" \
--partition_1 "./dataset/fetaqa_inc/train_1.json" \
--output_dir "./model/feta_LoRA_d1"
```

## LLM-based Query Generators

We train query generators for tabular data, based on the refined Llama3-8b model, as detailed in our paper. We release our trained [Query Generators](https://drive.google.com/drive/folders/1HLYbWzADI0xAqpuXuNgFcZPFxu0a2vfc?usp=drive_link). Users can also train their query generators for different table repositories.


## Acknowledgments

We thank the previous studies on table discovery/retrieval [Solo](https://github.com/thedatastation/solo), [DTR](https://github.com/google-research/tapas/blob/master/DENSE_TABLE_RETRIEVER.md). We use part of the code of [DSI-QG](https://github.com/ArvinZhuang/DSI-QG).
