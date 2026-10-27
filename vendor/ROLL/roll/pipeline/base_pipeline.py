import os
import re
import shutil
import stat
import time
from collections import defaultdict
from concurrent import futures
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import set_seed

from roll.distributed.executor.cluster import Cluster
from roll.distributed.executor.model_update_group import ModelUpdateGroup
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.scheduler.resource_manager import ResourceManager
from roll.utils.checkpoint_manager import CheckpointManager, download_model
from roll.utils.functionals import reduce_metrics
from roll.utils.logging import get_logger
from roll.utils.tracking import create_tracker
from roll.utils.worker_state import WorkerState

logger = get_logger()


class BasePipeline:
    ROLLING_CHECKPOINT_NAMESPACE = "rolling"
    model_update_groups: List[ModelUpdateGroup] = []
    checkpoint_clusters: List = []

    def __init__(self, pipeline_config):
        set_seed(seed=pipeline_config.seed)
        self.pipeline_config = pipeline_config
        self.resource_manager = ResourceManager(
            num_nodes=self.pipeline_config.num_nodes, num_gpus_per_node=self.pipeline_config.num_gpus_per_node
        )
        self.state = WorkerState()
        self.checkpoint_manager = CheckpointManager(checkpoint_config=self.pipeline_config.checkpoint_config)
        self.tracker = create_tracker(
            tracker_name=self.pipeline_config.track_with,
            config=self.pipeline_config.to_dict(),
            **self.pipeline_config.tracker_kwargs,
        )
        self.resume_from_checkpoint = False
        self.executor: futures.ThreadPoolExecutor = futures.ThreadPoolExecutor(max_workers=5)
        self.resume_futures = []

        manual_checkpoint_path = getattr(self.pipeline_config, "manual_checkpoint_path", None)
        if manual_checkpoint_path:
            logger.info(
                "Manual checkpoint trigger enabled. Create %s to save at the end of the current step.",
                os.path.abspath(os.path.expanduser(manual_checkpoint_path)),
            )

        if self.pipeline_config.resume_from_checkpoint:
            self.resume_from_checkpoint = download_model(self.pipeline_config.resume_from_checkpoint)

            logger.info(f"resume_from_checkpoint: {self.resume_from_checkpoint}")
            load_dir = os.path.join(self.resume_from_checkpoint, "pipeline")
            self.state = WorkerState.load_from_json(load_dir=load_dir, tag="pipeline")

            def resume_metrics():
                for metrics in self.state.log_history:
                    self.tracker.log(values=metrics, step=metrics["system/step"])

            self.resume_futures.append(self.executor.submit(resume_metrics))

    def run(self):
        pass

    def set_model_update_pair(self, src_cluster, tgt_cluster, frequency=1):
        self.model_update_groups.append(
            ModelUpdateGroup(src_cluster=src_cluster, tgt_cluster=tgt_cluster, frequency=frequency, pipeline_config=self.pipeline_config)
        )

    def set_checkpoint_clusters(self, *clusters):
        self.checkpoint_clusters.extend(clusters)

    def model_update(self, global_step):
        metrics = {}
        for model_update_group in self.model_update_groups:
            metrics.update(model_update_group.model_update(global_step))
            model_update_group.tgt_cluster.process_weights_after_loading()
        return metrics

    def _manual_checkpoint_request(self) -> Optional[Tuple[str, int, int, int, int]]:
        request_path = getattr(self.pipeline_config, "manual_checkpoint_path", None)
        if not request_path:
            return None

        request_path = os.path.abspath(os.path.expanduser(request_path))
        try:
            request_stat = os.stat(request_path)
        except FileNotFoundError:
            return None

        if not stat.S_ISREG(request_stat.st_mode):
            logger.warning("Ignoring manual checkpoint trigger because it is not a file: %s", request_path)
            return None

        return (
            request_path,
            request_stat.st_dev,
            request_stat.st_ino,
            request_stat.st_mtime_ns,
            request_stat.st_size,
        )

    @staticmethod
    def _consume_manual_checkpoint_request(request: Tuple[str, int, int, int, int]) -> None:
        request_path, *expected_signature = request
        try:
            request_stat = os.stat(request_path)
        except FileNotFoundError:
            return

        current_signature = [
            request_stat.st_dev,
            request_stat.st_ino,
            request_stat.st_mtime_ns,
            request_stat.st_size,
        ]
        if current_signature != expected_signature:
            logger.info("A newer manual checkpoint request is pending at %s", request_path)
            return

        try:
            os.unlink(request_path)
        except FileNotFoundError:
            pass

    def do_checkpoint(self, global_step, is_last_step=None):
        if is_last_step is None:
            is_last_step = global_step == self.pipeline_config.max_steps - 1

        metrics = self.state.log_history[-1]
        metrics["system/step"] = global_step
        # This hook runs after the optimizer update and pipeline state commit.
        # Step 0 is a completed training iteration, so explicit saves are valid.
        manual_request = self._manual_checkpoint_request() if global_step >= 0 else None
        scheduled_checkpoint = global_step > 0 and (
            global_step % self.pipeline_config.save_steps == 0 or global_step == self.pipeline_config.max_steps - 1
        )
        rolling_save_steps = int(getattr(self.pipeline_config, "rolling_save_steps", 0))
        rolling_checkpoint = (
            global_step > 0
            and rolling_save_steps > 0
            and global_step % rolling_save_steps == 0
        )
        if scheduled_checkpoint or rolling_checkpoint or manual_request is not None:
            # A regular scheduled/manual save wins when both cadences coincide.
            # The same full state would otherwise be serialized twice at that step.
            checkpoint_namespace = None
            if not scheduled_checkpoint and manual_request is None:
                checkpoint_namespace = self.ROLLING_CHECKPOINT_NAMESPACE
            if manual_request is not None:
                logger.info(
                    "Manual checkpoint requested via %s; saving checkpoint-%s",
                    manual_request[0],
                    global_step,
                )
            if checkpoint_namespace is not None:
                rolling_max_ckpt = int(
                    getattr(self.pipeline_config, "rolling_max_ckpt_to_keep", 1)
                )
                logger.info(
                    "Saving rolling checkpoint-%s; old checkpoints will be pruned only after the save succeeds",
                    global_step,
                )
            self._validate_checkpoint_step(global_step, namespace=checkpoint_namespace)
            # Keep every existing checkpoint through serialization and upload.
            # Insufficient space must never be resolved by deleting the fallback.
            ckpt_metrics_refss = []
            for cluster in self.checkpoint_clusters:
                checkpoint_kwargs = {
                    "global_step": global_step,
                    "is_last_step": is_last_step,
                    "blocking": False,
                }
                if checkpoint_namespace is not None:
                    checkpoint_kwargs["checkpoint_namespace"] = checkpoint_namespace
                ckpt_metrics_refss.append(cluster.do_checkpoint(**checkpoint_kwargs))

            for ckpt_metrics_refs in ckpt_metrics_refss:
                ckpt_metrics = DataProto.materialize_concat(data_refs=ckpt_metrics_refs)
                metrics.update(reduce_metrics(ckpt_metrics.meta_info.pop("metrics", {})))

            local_ckpt_id = f"checkpoint-{global_step}"
            ckpt_id = local_ckpt_id
            pipeline_path_parts = [self.pipeline_config.output_dir, "pipeline"]
            if checkpoint_namespace is not None:
                ckpt_id = f"{checkpoint_namespace}/{local_ckpt_id}"
                pipeline_path_parts.append(checkpoint_namespace)
            pipeline_path_parts.append(local_ckpt_id)
            pipeline_save_dir = os.path.join(*pipeline_path_parts)
            save_dir = os.path.join(pipeline_save_dir, "pipeline")
            self.state.save_to_json(save_dir=save_dir, tag="pipeline")
            self.state.save_rng_state(save_dir=save_dir, tag="pipeline")
            self.checkpoint_manager.upload(ckpt_id=ckpt_id, local_state_path=pipeline_save_dir)

            # Workers have finished, and pipeline state/RNG have been persisted.
            # Any save/upload exception above leaves all older checkpoints intact.
            if checkpoint_namespace is not None:
                self._cleanup_old_checkpoints(
                    strict=True,
                    namespace=checkpoint_namespace,
                    max_ckpt=rolling_max_ckpt,
                )
            else:
                # Clean up regular checkpoints according to their original,
                # independent retention policy.
                self._cleanup_old_checkpoints(
                    strict=getattr(
                        self.pipeline_config,
                        "checkpoint_pre_cleanup_enabled",
                        False,
                    )
                )

            metrics["system/checkpoint/saved"] = 1.0
            metrics["system/checkpoint/manual"] = float(manual_request is not None)
            metrics["system/checkpoint/rolling"] = float(checkpoint_namespace is not None)
            if manual_request is not None:
                self._consume_manual_checkpoint_request(manual_request)

        futures.wait(self.resume_futures)
        self.resume_futures.clear()

    def _checkpoint_storage_roots(self):
        """Return local staging and durable file-system checkpoint roots."""
        roots = []
        local_output_dir = getattr(self.pipeline_config, "output_dir", None)
        if local_output_dir:
            roots.append(os.path.realpath(local_output_dir))

        checkpoint_config = getattr(self.pipeline_config, "checkpoint_config", None) or {}
        if checkpoint_config.get("type") == "file_system" and (
            getattr(self.pipeline_config, "checkpoint_pre_cleanup_enabled", False)
            or int(getattr(self.pipeline_config, "rolling_save_steps", 0)) > 0
        ):
            durable_output_dir = checkpoint_config.get("output_dir")
            if durable_output_dir:
                roots.append(os.path.realpath(durable_output_dir))

        return list(dict.fromkeys(roots))



    def _checkpoint_dirs_by_step(self, namespace=None):
        """Index regular or namespaced durable/local checkpoint directories."""
        ckpt_pattern = re.compile(r"^checkpoint-(\d+)$")
        checkpoint_dirs = defaultdict(list)

        def index_children(parent):
            if not os.path.isdir(parent):
                return
            for checkpoint_name in os.listdir(parent):
                match = ckpt_pattern.match(checkpoint_name)
                if not match:
                    continue
                checkpoint_path = os.path.join(parent, checkpoint_name)
                if os.path.isdir(checkpoint_path):
                    checkpoint_dirs[int(match.group(1))].append(checkpoint_path)

        for root in self._checkpoint_storage_roots():
            if not os.path.isdir(root):
                continue
            if namespace is None:
                index_children(root)
                for item in os.listdir(root):
                    if item == self.ROLLING_CHECKPOINT_NAMESPACE:
                        continue
                    item_path = os.path.join(root, item)
                    if os.path.isdir(item_path):
                        index_children(item_path)
            else:
                index_children(os.path.join(root, namespace))
                for item in os.listdir(root):
                    if item == namespace:
                        continue
                    item_path = os.path.join(root, item, namespace)
                    index_children(item_path)

        return checkpoint_dirs

    @staticmethod
    def _delete_checkpoint_steps(checkpoint_dirs, steps_to_delete, strict):
        for step in steps_to_delete:
            for ckpt_dir in checkpoint_dirs[step]:
                if not os.path.exists(ckpt_dir):
                    continue
                try:
                    shutil.rmtree(ckpt_dir)
                    logger.info(f"Deleted old checkpoint: {ckpt_dir}")
                except Exception as e:
                    logger.warning(f"Failed to delete checkpoint {ckpt_dir}: {e}")
                    if strict:
                        raise

    def _validate_checkpoint_step(self, global_step, namespace=None):
        """Reject stale saves without deleting any existing checkpoint files."""
        checkpoint_dirs = self._checkpoint_dirs_by_step(namespace=namespace)
        future_steps = sorted(step for step in checkpoint_dirs if step > global_step)
        if future_steps:
            raise RuntimeError(
                f"checkpoint storage contains steps newer than {global_step}: {future_steps}"
            )

    def _cleanup_old_checkpoints(self, strict=False, namespace=None, max_ckpt=None):
        """Remove old local and durable checkpoints according to retention."""
        if max_ckpt is None:
            max_ckpt = int(getattr(self.pipeline_config, "max_ckpt_to_keep", 0))
        if max_ckpt <= 0:
            return

        checkpoint_dirs = self._checkpoint_dirs_by_step(namespace=namespace)
        if not checkpoint_dirs:
            return

        # Sort steps and determine which to delete
        sorted_steps = sorted(checkpoint_dirs, reverse=True)
        steps_to_delete = sorted_steps[max_ckpt:]

        if not steps_to_delete:
            return

        checkpoint_label = "rolling checkpoints" if namespace else "checkpoints"
        logger.info(
            f"Cleaning up old {checkpoint_label}. Keeping {max_ckpt}, deleting steps: {steps_to_delete}"
        )

        self._delete_checkpoint_steps(checkpoint_dirs, steps_to_delete, strict=strict)

    def download_models(self, *clusters: Cluster):
        node2pg: Dict[str, PlacementGroup] = {}
        node2model_names: Dict[str, set[str]] = defaultdict(set)
        for cluster in clusters:
            assert cluster.placement_groups is not None
            for pg_list in cluster.placement_groups:
                assert len(pg_list) > 0
                worker_nodes = set()
                for pg in pg_list:
                    node_rank = pg["node_rank"]
                    if node_rank not in worker_nodes:
                        worker_nodes.add(node_rank)
                        node2pg[node_rank] = pg["placement_group"]
                        if cluster.worker_config.model_args.model_name_or_path:
                            node2model_names[node_rank].add(cluster.worker_config.model_args.model_name_or_path)
                        if self.pipeline_config.resume_from_checkpoint:
                            node2model_names[node_rank].add(self.pipeline_config.resume_from_checkpoint)
        ray.get(
            [
                download_models.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=node2pg[node_rank])
                ).remote(model_name_or_paths=model_names)
                for node_rank, model_names in node2model_names.items()
            ]
        )

@ray.remote
def download_models(model_name_or_paths: set[str]):
    with futures.ThreadPoolExecutor(max_workers=5) as thread_executor:
        futures.wait([thread_executor.submit(download_model, model_name_or_path)
                      for model_name_or_path in model_name_or_paths])
