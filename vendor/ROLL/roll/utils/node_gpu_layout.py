"""Optional dense logical GPU ranks over explicitly selected, unequal nodes."""
import json
import os


def node_gpu_layout(value=None):
    value = os.environ.get("ROLL_NODE_GPU_LAYOUT", "") if value is None else value
    if not value:
        return []
    layout = json.loads(value) if isinstance(value, str) else value
    if not isinstance(layout, list) or not layout:
        raise ValueError("ROLL_NODE_GPU_LAYOUT must be a nonempty list")
    ips = set()
    for node in layout:
        ip, gpus = node["ip"], node["gpus"]
        if not isinstance(ip, str) or not ip or ip in ips:
            raise ValueError("GPU layout requires distinct node IPs")
        if (not isinstance(gpus, list) or not gpus or
                any(type(gpu) is not int or gpu < 0 for gpu in gpus) or
                len(set(gpus)) != len(gpus)):
            raise ValueError("GPU layout requires distinct nonnegative physical GPU indices")
        ips.add(ip)
    return layout


def logical_gpu_map(layout):
    return [(node_rank, gpu) for node_rank, node in enumerate(layout) for gpu in node["gpus"]]
