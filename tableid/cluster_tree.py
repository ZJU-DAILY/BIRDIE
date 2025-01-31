import numpy as np
import pickle
import argparse
import os
import random
from sklearn.preprocessing import normalize
import pandas as pd
import json
import math


class cluster_node:
    def __init__(self, node_id, father, center, radius, cohesion):
        self.table_ids = []
        self.children = []
        self.outlier = []
        self.father = father
        self.node_id = node_id
        self.center = center  # cluster center
        self.radius = radius  # cluster radius
        self.cohesion = cohesion  # cohesion

    def add_child(self, child_node):
        if isinstance(child_node, list):
            self.children.extend(child_node)
        else:
            self.children.append(child_node)

    def add_table(self, table_ids):
        if isinstance(table_ids, list):
            self.table_ids.extend(table_ids)
        else:
            self.table_ids.append(table_ids)

    def __repr__(self):
        return (
            f"cluster_node(node_id={self.node_id}, center={self.center}, "
            f"radius={self.radius}, cohesion={self.cohesion}, "
            f"table_ids={self.table_ids}, children_count={len(self.children)})"
        )


class cluster_tree:
    def __init__(self, tree_path):
        self.root = self.load_tree(tree_path)
        self.level_averages = self.get_cluster_metrics()

    def load_tree(self, tree_path):
        with open(tree_path, "rb") as f:
            tree = pickle.load(f)
        return tree

    def update_tree(self, emd1, emd2):
        new_ids = []

        for data_point in range(len(emd1)):
            if self.root.children:
                nearest_child = self.root.children[0]
                for child in self.root.children:
                    if np.linalg.norm(child.center - emd1[data_point]) < np.linalg.norm(
                        nearest_child.center - emd1[data_point]
                    ):
                        nearest_child = child
            new_ids.append(
                self._insert_into_tree(nearest_child, data_point, emd1, emd2, 1)
            )
        return new_ids

    def _insert_into_tree(self, node, data_point, emd1, emd2, indicate):
        emd = emd2 if indicate > 2 else emd1
        d = np.linalg.norm(node.center - emd[data_point])
        if d <= node.cohesion:
            table_id = len(node.table_ids)
            node.add_table(data_point)
            if node.children:
                nearest_child = node.children[0]
                for child in node.children:
                    if np.linalg.norm(child.center - emd[data_point]) < np.linalg.norm(
                        nearest_child.center - emd[data_point]
                    ):
                        nearest_child = child
                new_id = self._insert_into_tree(
                    nearest_child, data_point, emd1, emd2, indicate + 1
                )
                if new_id:
                    return [node.node_id] + new_id
            return [node.node_id] + [table_id]

        elif d <= node.radius:
            table_id = len(node.table_ids)
            old_center = node.center
            node.center = np.mean(
                [emd[data_point]] + [node.center for _ in node.table_ids], axis=0
            )
            new_d = np.linalg.norm(node.center - emd[data_point])
            node.cohesion = np.mean(new_d + [node.cohesion for _ in node.table_ids])
            new_radius_min = np.linalg.norm(emd[data_point] - node.center)
            new_radius_max = d + np.linalg.norm(node.center - old_center)
            node.radius = random.uniform(new_radius_min, new_radius_max)

            if math.isnan(node.radius) or node.radius <= 0:
                raise ValueError(f"Invalid avg_radius: {node.radius}")

            if math.isnan(node.cohesion) or node.cohesion <= 0:
                raise ValueError(f"Invalid avg_cohesion: {node.cohesion}")

            node.add_table(data_point)
            if node.children:
                nearest_child = node.children[0]
                for child in node.children:
                    if np.linalg.norm(child.center - emd[data_point]) < np.linalg.norm(
                        nearest_child.center - emd[data_point]
                    ):
                        nearest_child = child
                new_id = self._insert_into_tree(
                    nearest_child, data_point, emd1, emd2, indicate + 1
                )
                if new_id:
                    return [node.node_id] + new_id
            return [node.node_id] + [table_id]

        else:
            node_radius, node_cohesion = self.average_radius_and_cohesion(indicate)
            new_node = cluster_node(
                node_id=len(node.father.children),
                father=node.father,
                center=emd[data_point],
                radius=node_radius,
                cohesion=node_cohesion,
            )
            new_node.add_table(data_point)
            node.father.add_child(new_node)
            return [new_node.node_id] + [len(new_node.table_ids) - 1]

    def get_cluster_metrics(self):
        metrics = []
        self._get_cluster_metrics_recursive(self.root, metrics, 0)
        level_averages = {}
        for level in range(len(metrics)):
            level_data = np.array(metrics[level])
            avg_radius = np.mean(level_data[:, 0])
            avg_cohesion = np.mean(level_data[:, 1])
            level_averages[level] = (avg_radius, avg_cohesion)
        return level_averages

    def _get_cluster_metrics_recursive(self, node, metrics, level):
        if len(metrics) <= level:
            metrics.append([])
        metrics[level].append((node.radius, node.cohesion))
        for child in node.children:
            self._get_cluster_metrics_recursive(child, metrics, level + 1)

    def print_leaf_paths(self):
        paths = []
        self._traverse_and_collect_paths(self.root, [], paths)
        for path in paths:
            print(" -> ".join(map(str, path[1:])))

    def _traverse_and_collect_paths(self, node, current_path, paths):
        if math.isnan(node.radius) or node.radius < 0:
            raise ValueError(f"Invalid avg_radius: {node.radius}")

        if math.isnan(node.cohesion) or node.cohesion < 0:
            raise ValueError(f"Invalid avg_cohesion: {node.cohesion}")

        current_path.append(node.node_id)
        if not node.children:
            paths.append(current_path.copy())
        else:
            for child in node.children:
                self._traverse_and_collect_paths(child, current_path, paths)
        if current_path:
            current_path.pop()

    def average_radius_and_cohesion(self, indicate):
        avg_radius = self.level_averages[indicate][0]
        avg_cohesion = self.level_averages[indicate][1]

        if math.isnan(avg_radius) or avg_radius < 0:
            raise ValueError(f"Invalid avg_radius: {avg_radius}")

        if math.isnan(avg_cohesion) or avg_cohesion < 0:
            raise ValueError(f"Invalid avg_cohesion: {avg_cohesion}")
        return avg_radius, avg_cohesion


def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="BIRDIE/tableid/temp")
    parser.add_argument("--dataset_name", default="fetaqa_inc_1", type=str)
    parser.add_argument("--base_tag", type=str, default="fetaqa_inc_0")
    parser.add_argument("--semantic_id_dir", type=str, default="BIRDIE/tableid/docid/")

    return parser.parse_args()

if __name__ == "__main__":
    args = create_args()
    data_path = os.path.join(args.data_path, args.dataset_name)
    tree_path = os.path.join(
        data_path, f"hierarchical_clustering_tree_{args.dataset_name}.pkl"
    )
    table_title_schema_embedding = np.load(
        os.path.join(data_path, f"table_title_schema_embedding_{args.base_tag}.npy")
    )
    table_data_embedding = np.load(
        os.path.join(data_path, f"table_data_embedding_{args.base_tag}.npy")
    )

    table_title_schema_embedding = normalize(table_title_schema_embedding, norm="l2")
    table_data_embedding = normalize(table_data_embedding, norm="l2")

    tree = cluster_tree(tree_path)

    print(tree.get_cluster_metrics())

    tree.print_leaf_paths()
    new_docid = tree.update_tree(table_title_schema_embedding, table_data_embedding)

    tree.print_leaf_paths()

    string_semantic_id = [
        "".join([str(x).zfill(2) for x in new_docid[i]]) for i in range(len(new_docid))
    ]

    origin_length = len(
        set(["".join([str(x) for x in new_docid[i]]) for i in range(len(new_docid))])
    )
    final_length = len(set(string_semantic_id))
    print(
        f"new_id length: {origin_length}.  semantic_id_length: {final_length} dataset_length {len(new_docid)}"
    )
    id_map = []

    df = pd.read_csv(os.path.join(data_path, f"tableId_list_{args.base_tag}.csv"))
    tableId_list_loaded = list(df.itertuples(index=False, name=None))
    old_docid = [x[0] for x in tableId_list_loaded]
    docid_map = None
    docid_emd_map = {
        tableId_list[0]: tableId_list[1] for tableId_list in tableId_list_loaded
    }

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
    with open(os.path.join(id_output_path, f"id_map_{args.dataset_name}.json"), "w") as tf:
        [tf.write(json.dumps(item) + "\n") for item in id_map]

    with open(
        os.path.join(data_path, f"hierarchical_clustering_tree_{args.dataset_name}.pkl"),
        "wb",
    ) as f:
        pickle.dump(tree.root, f)
