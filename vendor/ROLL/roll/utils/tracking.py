import json
from functools import wraps
from typing import Optional, Dict, Any

import torch

from roll.utils.logging import get_logger

logger = get_logger()

tracker_registry: Dict[str, Any] = {}


def _strip_metric_tag(values: Dict[str, Any]) -> Dict[str, Any]:
    """
    Strip reducer tags from metric keys before logging.

    We may annotate metric keys with reducer tags for internal aggregation:
      "actor/pg_loss@sum", "actor/kl_loss@mean", ...
    Dashboards (TensorBoard/W&B/...) should log clean names, so we remove "@...":
      "actor/pg_loss@sum" -> "actor/pg_loss"

    - Only strips the last "@tag" part (rsplit("@", 1))
    - Recursively strips nested dict keys (e.g. add_scalars)
    - Returns a new dict (does not mutate the input)
    """
    def strip_key(k: str) -> str:
        return k.rsplit("@", 1)[0] if isinstance(k, str) and "@" in k else k

    out: Dict[str, Any] = {}
    for k, v in values.items():
        nk = strip_key(k)
        if isinstance(v, dict):
            v = _strip_metric_tag(v)
        out[nk] = v
    return out


def strip_at_tag_in_log(func):
    """
    Decorator for Tracker.log(...).

    Purpose:
      Remove "@tag" suffixes from metric keys right before sending them to the
      logging backend. This is name-cleaning only (no reduction happens here).
    """
    @wraps(func)
    def wrapper(self, values: dict, step: Optional[int] = None, **kwargs):
        if isinstance(values, dict):
            values = _strip_metric_tag(values)
        return func(self, values, step, **kwargs)
    return wrapper



class BaseTracker:

    def log(self, values: dict, step: Optional[int], **kwargs):
        pass

    def finish(self):
        pass








class StdoutTracker(BaseTracker):

    def __init__(self, config: dict, **kwargs):
        self.config = config

    @strip_at_tag_in_log
    def log(self, values: dict, step: Optional[int], **kwargs):
        logger.info(f"metrics_tag: {json.dumps({'step': step, 'metrics': values})}")

    def finish(self):
        pass


def create_tracker(tracker_name: str, config: dict, **kwargs) -> BaseTracker:
    if not tracker_name:
        return BaseTracker()
    logger.info(f"create tracker {tracker_name}, kwargs: {kwargs}")

    if tracker_name not in tracker_registry:
        raise ValueError(f"Unknown tracker name: {tracker_name}, total registered trackers: {tracker_registry.keys()}")
    tracker_cls = tracker_registry[tracker_name]
    return tracker_cls(config, **kwargs)


tracker_registry["stdout"] = StdoutTracker
