from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import pandas as pd
import pickle
import argparse
import os
import json
import pickle
from sklearn.preprocessing import normalize, LabelEncoder
from sklearnex import patch_sklearn
import math
from cluster_tree import cluster_node

patch_sklearn()


def find_root(node):
    while node.father is not None:
        node = node.father
    return node


def collect_nodes_at_level(root, level, current_level=0):
    if current_level == level:
        return [root]

    nodes_at_level = []
    for child in root.children:
        nodes_at_level.extend(collect_nodes_at_level(child, level, current_level + 1))

    return nodes_at_level


def average_radius_and_cohesion(node, level):
    root = find_root(node)
    nodes_at_same_level = collect_nodes_at_level(root, level)

    if not nodes_at_same_level:
        return node.radius, node.cohesion

    radii = [n.radius for n in nodes_at_same_level if n.radius is not None]
    cohesions = [n.cohesion for n in nodes_at_same_level if n.cohesion is not None]

    avg_radius = np.nanmean(radii) if radii else node.radius
    avg_cohesion = np.nanmean(cohesions) if cohesions else node.cohesion

    avg_radius = avg_radius if avg_radius > 0 else node.radius
    avg_cohesion = avg_cohesion if avg_cohesion > 0 else node.cohesion

    if math.isnan(avg_radius) or avg_radius <= 0:
        raise ValueError(f"Invalid avg_radius: {avg_radius}")

    if math.isnan(avg_cohesion) or avg_cohesion <= 0:
        raise ValueError(f"Invalid avg_cohesion: {avg_cohesion}")

    return avg_radius, avg_cohesion


def cluster_recursion(
    x_data_pos,
    new_docid,
    indicate,
    kmeans,
    mini_kmeans,
    emd1,
    emd2,
    father_node: cluster_node,
):
    emd = emd2 if indicate >= 2 else emd1

    if x_data_pos.shape[0] <= args.c:
        for idx, pos in enumerate(x_data_pos):
            new_docid[pos].append(idx)
        return

    if x_data_pos.shape[0] >= 1e3:
        pred = mini_kmeans.fit_predict(emd[x_data_pos])
    else:
        pred = kmeans.fit_predict(emd[x_data_pos])

    pred = LabelEncoder().fit_transform(pred)
    uni_clusters = [int(i) for i in np.unique(pred)]
    if len(uni_clusters) == 1:
        for idx, pos in enumerate(x_data_pos):
            new_docid[pos].append(idx)
        return

    for i in uni_clusters:
        pos_lists = []
        for id_, class_ in enumerate(pred):
            if class_ == i:
                pos_lists.append(x_data_pos[id_])
                new_docid[x_data_pos[id_]].append(i)
        data_pos = np.array(pos_lists)
        if 0 < len(pos_lists) <= args.c:
            if len(pos_lists) == 1:
                node_radius, node_cohesion = average_radius_and_cohesion(
                    father_node, indicate + 1
                )
                node = cluster_node(
                    node_id=i,
                    father=father_node,
                    center=emd[data_pos[0]],
                    radius=node_radius,
                    cohesion=node_cohesion,
                )

                node.add_table(pos_lists)
                father_node.add_child(node)
                cluster_recursion(
                    data_pos,
                    new_docid,
                    indicate + 1,
                    kmeans,
                    mini_kmeans,
                    emd1,
                    emd2,
                    node,
                )
            else:
                node_distance = np.linalg.norm(
                    emd[data_pos] - np.mean(emd[data_pos], axis=0), axis=1
                )
                node = cluster_node(
                    node_id=i,
                    father=father_node,
                    center=np.mean(emd[data_pos], axis=0),
                    radius=np.max(node_distance),
                    cohesion=np.mean(node_distance),
                )

                node.add_table(pos_lists)
                father_node.add_child(node)
                cluster_recursion(
                    data_pos,
                    new_docid,
                    indicate + 1,
                    kmeans,
                    mini_kmeans,
                    emd1,
                    emd2,
                    node,
                )
        elif len(pos_lists) > args.c:
            node_distance = np.linalg.norm(
                emd[data_pos] - np.mean(emd[data_pos], axis=0), axis=1
            )
            node = cluster_node(
                node_id=i,
                father=father_node,
                center=np.mean(emd[data_pos], axis=0),
                radius=np.max(node_distance),
                cohesion=np.mean(node_distance),
            )
            node.add_table(pos_lists)
            father_node.add_child(node)
            cluster_recursion(
                data_pos, new_docid, indicate + 1, kmeans, mini_kmeans, emd1, emd2, node
            )

    return


def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v_dim", type=int, default=768)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--k", type=int, default=20
    )  # 32 for nq_tables      20 for fetaqa
    parser.add_argument(
        "--c", type=int, default=20
    )  # 32 for nq_tables      20 for fetaqa
    parser.add_argument("--data_path", type=str, default="BIRDIE/tableid/temp")
    parser.add_argument("--dataset_name", default="fetaqa", type=str)
    parser.add_argument("--semantic_id_dir", type=str, default="BIRDIE/tableid/docid/")
    return parser.parse_args()


if __name__ == "__main__":
    args = create_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

    run_tag = args.dataset_name

    tableId_list_loaded = []
    data_path = os.path.join(args.data_path, args.dataset_name)
    table_title_schema_embedding = np.load(
        os.path.join(data_path, f"table_title_schema_embedding_{run_tag}.npy")
    )
    table_data_embedding = np.load(
        os.path.join(data_path, f"table_data_embedding_{run_tag}.npy")
    )

    df = pd.read_csv(os.path.join(data_path, f"tableId_list_{run_tag}.csv"))
    tableId_list_loaded = list(df.itertuples(index=False, name=None))

    table_title_schema_embedding = normalize(table_title_schema_embedding, norm="l2")
    table_data_embedding = normalize(table_data_embedding, norm="l2")

    tableId_list_loaded.extend(tableId_list_loaded)

    old_docid = [x[0] for x in tableId_list_loaded]
    docid_map = None
    docid_emd_map = {
        tableId_list[0]: tableId_list[1] for tableId_list in tableId_list_loaded
    }

    new_docid = []
    for _ in range(len(table_title_schema_embedding)):
        new_docid.append([])
    kmeans = KMeans(
        n_clusters=args.k,
        max_iter=300,
        n_init=100,
        init="k-means++",
        random_state=args.seed,
        tol=1e-7,
    )

    mini_kmeans = MiniBatchKMeans(
        n_clusters=args.k,
        max_iter=300,
        n_init=100,
        init="k-means++",
        random_state=args.seed,
        batch_size=1000,
        reassignment_ratio=0.01,
        max_no_improvement=20,
        tol=1e-7,
    )

    root_center = np.mean(table_title_schema_embedding, axis=0)
    root_radius = np.max(
        np.linalg.norm(table_title_schema_embedding - root_center, axis=1)
    )
    root_cohesion = np.mean(
        np.linalg.norm(table_title_schema_embedding - root_center, axis=1)
    )
    root_node = cluster_node(
        node_id=0,
        father=None,
        center=root_center,
        radius=root_radius,
        cohesion=root_cohesion,
    )

    cluster_recursion(
        np.array(range(len(table_title_schema_embedding))),
        new_docid,
        0,
        kmeans,
        mini_kmeans,
        table_title_schema_embedding,
        table_data_embedding,
        root_node,
    )

    string_semantic_id = [
        "".join([str(x).zfill(2) for x in new_docid[i]]) for i in range(len(new_docid))
    ]

    origin_length = len(
        set(["".join([str(x) for x in new_docid[i]]) for i in range(len(new_docid))])
    )
    final_length = len(set(string_semantic_id))

    # print(f"new_id length: {origin_length}.  semantic_id_length: {final_length} dataset_length {len(new_docid)}")
    id_map = []

    for i in range(len(string_semantic_id)):
        id_map.append(
            {
                "text_id": old_docid[i],
                "tableID": docid_emd_map[old_docid[i]],
                "semantic_id": string_semantic_id[i],
                "semantic_id_list": new_docid[i],
            }
        )
    id_output_path = os.path.join(args.semantic_id_dir, args.dataset_name)
    os.makedirs(id_output_path, exist_ok=True)
    with open(os.path.join(id_output_path, f"id_map_{run_tag}.json"), "w") as tf:
        [tf.write(json.dumps(item) + "\n") for item in id_map]

    print("Saving hierarchical clustering tree...")
    with open(
        os.path.join(data_path, f"hierarchical_clustering_tree_{run_tag}.pkl"), "wb"
    ) as f:
        pickle.dump(root_node, f)
