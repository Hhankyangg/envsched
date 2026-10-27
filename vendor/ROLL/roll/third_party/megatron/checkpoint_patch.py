"""Keep fully-reshardable optimizer offsets consistent with coalesced buckets.

Megatron's DP-zero export drops padding between buckets, while its fully
reshardable exporter indexes that result using padded buffer offsets. Translate
only the exporter's index maps; live training buffers and DP shard maps retain
their original coordinates.
"""
from copy import copy
from functools import wraps
import logging
import os


def validate_exported_optimizer(buffers, state):
    """Check the actual export after gathering, before any checkpoint is written."""
    import torch
    if state is None:
        return
    index = elements = 0
    for buffer in buffers:
        for model in buffer.param_index_map:
            exported = state[index]
            master = exported["param"].data.detach().reshape(-1)
            reference = model.detach().reshape(-1)
            if master.numel() != reference.numel():
                raise RuntimeError("Exported optimizer/model parameter sizes differ")
            for start in range(0, master.numel(), 1048576):
                actual = master[start:start+1048576]
                expected = reference[start:start+1048576].to(device=actual.device)
                if not torch.equal(actual.to(expected.dtype), expected):
                    raise RuntimeError(f"Optimizer export differs from model: {exported['param'].key}")
                for name in ("param", "exp_avg", "exp_avg_sq"):
                    chunk = exported[name].data.detach().reshape(-1)[start:start+1048576]
                    if not torch.isfinite(chunk).all() or (name == "exp_avg_sq" and (chunk < 0).any()):
                        raise RuntimeError(f"Invalid optimizer export: {exported[name].key}")
            elements += master.numel()
            index += 1
    logging.getLogger(__name__).info("Optimizer export verified: %d parameters, %d elements", index, elements)


def coalesced_param_index_map(buffer):
    starts = []
    total = 0
    for bucket in buffer.buckets:
        starts.append(total)
        total += bucket.numel_unpadded
    if total != buffer.numel_unpadded:
        raise RuntimeError("Optimizer buffer and bucket unpadded sizes disagree")
    result = {}
    for param, (start, end, bucket_id) in buffer.param_index_map.items():
        bucket = buffer.buckets[bucket_id]
        local_start, local_end = start - bucket.offset, end - bucket.offset
        if not 0 <= local_start <= local_end <= bucket.numel_unpadded:
            raise RuntimeError("Optimizer parameter lies outside its unpadded bucket")
        result[param] = (starts[bucket_id] + local_start,
                         starts[bucket_id] + local_end, bucket_id)
    return result


def bind_coalesced_checkpoint_export(optimizer):
    """Bind once per optimizer; do not alter an installed Megatron package."""
    if getattr(optimizer, "_roll_coalesced_checkpoint_export", False):
        return
    original = getattr(optimizer, "sharded_param_state_fully_reshardable", None)
    if original is None:
        return

    @wraps(original)
    def export(*args, **kwargs):
        original_buffers = optimizer.buffers
        export_buffers = []
        for buffer in original_buffers:
            proxy = copy(buffer)
            proxy.param_index_map = coalesced_param_index_map(buffer)
            export_buffers.append(proxy)
        optimizer.buffers = export_buffers
        try:
            result = original(*args, **kwargs)
            is_loading = kwargs.get("is_loading", args[1] if len(args) > 1 else False)
            if not is_loading and os.environ.get("ROLL_VERIFY_CHECKPOINT_EXPORT") == "1":
                validate_exported_optimizer(original_buffers, result)
            return result
        finally:
            optimizer.buffers = original_buffers

    optimizer.sharded_param_state_fully_reshardable = export
    optimizer._roll_coalesced_checkpoint_export = True
