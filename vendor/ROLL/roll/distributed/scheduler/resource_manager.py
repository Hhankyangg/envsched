import dataclasses
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import ray
from ray.util.placement_group import PlacementGroup

from roll.platforms import current_platform
from roll.utils.ray_utils import get_visible_gpus, get_node_rank
from roll.utils.node_gpu_layout import node_gpu_layout, logical_gpu_map


class ResourceManager:
    def __init__(self, num_gpus_per_node, num_nodes):
        """
            The ResourceManager centrally manages the required GPU/CPU resources,
            facilitating Ray to deploy Actors on specified GPU devices.
        """
        layout = node_gpu_layout()
        if layout:
            self._initialize_explicit_layout(layout)
            return
        self.logical_gpu_map = None
        available_resources = ray.available_resources()
        available_gpu = available_resources.get(current_platform.ray_device_key, 0)

        nodes_maybe_used = []
        ray_nodes = ray.nodes()
        for node in ray_nodes:
            resource = node["Resources"]
            node_gpu_num = int(resource.get(current_platform.ray_device_key, 0))
            if node_gpu_num >= num_gpus_per_node:
                nodes_maybe_used.append(node)
        nodes_maybe_used = sorted(nodes_maybe_used, key=lambda n: n["Resources"]["CPU"])

        ray_num_nodes = len(nodes_maybe_used)
        if num_nodes is None:
            num_nodes = ray_num_nodes

        assert num_nodes <= ray_num_nodes, (f"The Ray clusters(ray_num_nodes: {ray_num_nodes}) cannot meet the "
                                            f"required number of nodes (`num_nodes`{num_nodes}).")
        self.num_nodes = num_nodes
        self.gpu_per_node = num_gpus_per_node
        self.num_gpus = self.gpu_per_node * self.num_nodes

        if self.gpu_per_node > 0:
            assert self.num_gpus <= available_gpu, f"num_gpus {self.num_gpus} > available_gpu {available_gpu}"
            bundles = []
            for i in range(self.num_nodes):
                node = nodes_maybe_used[i]
                node_cpu = int(node["Resources"]["CPU"])
                bundles.append({current_platform.ray_device_key: self.gpu_per_node, "CPU": max(node_cpu / 2, 1)})

            self.placement_groups = [ray.util.placement_group([bundle]) for bundle in bundles]
            ray.get([pg.ready() for pg in self.placement_groups])
            gpu_ranks = ray.get([
                get_visible_gpus.options(
                    placement_group=pg,
                    **(
                        {"num_gpus": self.gpu_per_node}
                        if current_platform.ray_device_key == "GPU"
                        else {"resources": {current_platform.ray_device_key: self.gpu_per_node}}
                    )
                ).remote(current_platform.device_control_env_var)
                for pg in self.placement_groups
            ])
            print(f"gpu ranks: {gpu_ranks}")
            self.node_ranks = ray.get(
                [get_node_rank.options(placement_group=pg).remote() for pg in self.placement_groups])
            if self.node_ranks.count(0) > 1:
                # NODE_RANK environment variable is not set in the cluster, so a default value is used for NODE_RANK.
                self.node_ranks = list(range(len(self.placement_groups)))

            self.gpu_ranks = [int(gpu_rank[0]) for gpu_rank in gpu_ranks]
            self.node2pg: Dict[int, PlacementGroup] = {}
            for node_rank, placement_group in zip(self.node_ranks, self.placement_groups):
                self.node2pg[node_rank] = placement_group
            print(f"node2pg: {self.node2pg}")
        else:
            assert self.num_nodes == 1
            node = nodes_maybe_used[0]
            node_cpu = int(node["Resources"]["CPU"])
            bundles = [{"CPU": node_cpu}] * self.num_nodes
            self.placement_groups = [ray.util.placement_group([bundle]) for bundle in bundles]
            ray.get([pg.ready() for pg in self.placement_groups])
            self.node_ranks = [0]
            self.node2pg: Dict[int, PlacementGroup] = {}
            for node_rank, placement_group in zip(self.node_ranks, self.placement_groups):
                self.node2pg[node_rank] = placement_group

    def _initialize_explicit_layout(self, layout):
        nodes = {node["NodeManagerAddress"]: node for node in ray.nodes() if node["Alive"]}
        self.num_nodes = len(layout)
        self.gpu_per_node = max(len(node["gpus"]) for node in layout)
        self.logical_gpu_map = logical_gpu_map(layout)
        self.num_gpus = len(self.logical_gpu_map)
        bundles = []
        for entry in layout:
            node = nodes.get(entry["ip"])
            if node is None:
                raise ValueError(f"Missing live Ray node {entry['ip']}")
            resources = node["Resources"]
            count = len(entry["gpus"])
            if resources.get(current_platform.ray_device_key, 0) != count:
                raise ValueError(f"Ray node {entry['ip']} must advertise exactly {count} selected GPUs")
            bundles.append({current_platform.ray_device_key: count,
                            "CPU": max(resources["CPU"] / 2, 1),
                            f"node:{entry['ip']}": 0.001})
        self.placement_groups = [ray.util.placement_group([bundle]) for bundle in bundles]
        try:
            ray.get([pg.ready() for pg in self.placement_groups], timeout=120)
            visible = ray.get([
                get_visible_gpus.options(placement_group=pg, num_gpus=len(entry["gpus"])).remote(
                    current_platform.device_control_env_var)
                for entry, pg in zip(layout, self.placement_groups)
            ], timeout=120)
            for entry, actual in zip(layout, visible):
                if set(map(int, actual)) != set(entry["gpus"]):
                    raise ValueError(f"Ray visible GPUs {actual} differ from layout {entry}")
        except Exception:
            self.destroy_placement_group()
            raise
        self.node_ranks = list(range(self.num_nodes))
        self.gpu_ranks = [entry["gpus"][0] for entry in layout]
        self.node2pg = dict(zip(self.node_ranks, self.placement_groups))
        print(f"Explicit logical GPU layout: {self.logical_gpu_map}")

    def nodes_placement_group(self, node_rank) -> PlacementGroup:
        """
        mesh table是 m×n，获取第node_rank nodel上gpu_rank的PlacementGroup，用于把ray.Actor部署到指定的GPU上
        """
        return self.node2pg[node_rank]

    def destroy_placement_group(self):
        for pg in self.placement_groups:
            ray.util.remove_placement_group(pg)

    def allocate_placement_group(self, world_size, device_mapping: List[int] = None) -> List[List[Dict]]:
        """
            Allocate resources according to device_mapping (numbered by GPU RANK)
            - GPUs: Specify required GPU indices via device_mapping
            - CPUs: Specify via world_size

            Return Type: List[List[Dict]]
              Dict Keys:
                - node_rank
                - gpu_rank
                - placement_group
              List[Dict]: Represents GPUs allocated to a worker and access to placement groups
              Example: If num_gpus_per_worker=8, then len(List[Dict])=8

            A Worker is defined as a group of resource owners (can span multiple machines) that can independently use allocated resources to execute computation operations.
        """
        allocated_pg = []
        ray_address = f"{ray.get_runtime_context().gcs_address}"
        if device_mapping:
            if world_size <= 0 or len(device_mapping) % world_size:
                raise ValueError("device_mapping must divide evenly over workers")
            num_gpus_per_worker = len(device_mapping) // world_size
            grouped_ranks = [
                list(device_mapping[i : i + num_gpus_per_worker])
                for i in range(0, len(device_mapping), num_gpus_per_worker)
            ]
            for group in grouped_ranks:
                pg_list = []
                for rank in group:
                    if self.logical_gpu_map is not None:
                        if not 0 <= rank < self.num_gpus:
                            raise ValueError(f"GPU rank {rank} is outside explicit layout")
                        node_rank, gpu_rank = self.logical_gpu_map[rank]
                    else:
                        node_rank = rank // self.gpu_per_node
                        gpu_rank = rank % self.gpu_per_node

                    assert node_rank < self.num_nodes, (f"device_mapping used gpus are more than "
                                                        f"num_nodes×num_gpus_per_node={self.num_nodes}×{self.gpu_per_node}")

                    pg = self.nodes_placement_group(node_rank)
                    pg_list.append(
                        dict(node_rank=node_rank, gpu_rank=gpu_rank, global_gpu_rank=rank,
                             placement_group=pg, ray_address=ray_address)
                    )
                allocated_pg.append(pg_list)
        else:
            # Try to spread the CPU workers across various nodes to avoid the out-of-memory (OOM) situation caused
            # by the concentration of CPU workers in one place and the resulting peak memory usage.
            for rank in range(world_size):
                node_rank = rank % self.num_nodes
                allocated_pg.append(
                    [
                        dict(
                            node_rank=node_rank,
                            gpu_rank=None,
                            placement_group=self.nodes_placement_group(node_rank),
                            ray_address=ray_address,
                        )
                    ]
                )

        assert len(allocated_pg) == world_size

        return allocated_pg
