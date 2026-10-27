import os
import json
import numpy
from codetiming import Timer
from roll.utils.logging import get_logger

logger = get_logger()

def write_to_json_process(path, data, columns_configs):
    os.makedirs(path, exist_ok=True)
    column_names = {item[0] for item in columns_configs}
    data = {k: v.tolist() if isinstance(v, numpy.ndarray) else v for k,v in data.items() if k in column_names}
    with Timer(name="dump", logger=None) as timer:
        global_step = data.get('global_step', [0])[0]
        with open(os.path.join(path, f"rollout_dump_data.step_{global_step}.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    logger.info(f"dump_rollout to {path}: {timer.last}")

def json_checker(path:str):
    return path.startswith("/")

DUMPING_FUNC = [
    [json_checker, write_to_json_process],
]

