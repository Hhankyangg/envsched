"""Check resumed optimizer master weights against their model shards."""
import torch


def optimizer_shards(optimizer):
    for opt in getattr(optimizer, "chained_optimizers", [optimizer]):
        for model_param in opt.model_param_group_index_map:
            span = opt._get_model_param_range_map(model_param)["param"]
            model = model_param.detach().view(-1)[span.start:span.end]
            states = opt._get_main_param_and_optimizer_states(model_param)
            yield model, states


@torch.no_grad()
def audit_master_weights(optimizer):
    count = elements = mismatch = nonfinite = negative_variance = 0
    for model, states in optimizer_shards(optimizer):
        master = states["param"].detach().view(-1)
        if model.numel() != master.numel():
            raise RuntimeError("Resume optimizer/model shard sizes differ")
        count += 1
        elements += model.numel()
        for start in range(0, model.numel(), 1048576):
            expected = master[start:start+1048576].to(device=model.device, dtype=model.dtype)
            mismatch += int((model[start:start+1048576] != expected).sum().item())
        for name in ("param", "exp_avg", "exp_avg_sq"):
            tensor = states[name].detach().view(-1)
            for start in range(0, tensor.numel(), 1048576):
                chunk = tensor[start:start+1048576]
                nonfinite += int((~torch.isfinite(chunk)).sum().item())
                if name == "exp_avg_sq":
                    negative_variance += int((chunk < 0).sum().item())
    result = dict(shards=count, elements=elements, cast_mismatches=mismatch,
                  nonfinite=nonfinite, negative_variance=negative_variance)
    if mismatch or nonfinite or negative_variance:
        raise RuntimeError(f"Invalid resumed optimizer/model state: {result}")
    return result


@torch.no_grad()
def snapshot_master_samples(optimizer):
    result = []
    for _, states in optimizer_shards(optimizer):
        master = states["param"].detach().view(-1)
        if master.numel():
            count = min(257, master.numel())
            # Float32 linspace can round numel-1 up to numel on large shards,
            # poisoning CUDA with an out-of-bounds gather before optimizer.step.
            # Keep the entire index calculation integral, including endpoints.
            indices = torch.arange(count, dtype=torch.int64, device=master.device)
            if count > 1:
                indices = torch.div(indices * (master.numel() - 1), count - 1,
                                    rounding_mode="floor")
            result.append((master, indices, master[indices].clone()))
    return result


@torch.no_grad()
def check_master_update(samples, limit=0.01):
    maximum = max((float((master[indices]-before).abs().max().item())
                   for master, indices, before in samples), default=0.0)
    if not torch.isfinite(torch.tensor(maximum)) or maximum > limit:
        raise RuntimeError(f"Unexpected resumed master-weight update: max_delta={maximum}")
    return maximum
