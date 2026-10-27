from roll.utils.asyncio_decorator import create_sync_class


def create_strategy(worker, sync_wrapper=False):
    name = worker.worker_config.strategy_args.strategy_name
    if name == "vllm":
        from roll.distributed.strategy.vllm_strategy import VllmStrategy as strategy_cls
    elif name == "megatron_train":
        from roll.distributed.strategy.megatron_strategy import MegatronTrainStrategy as strategy_cls
    elif name == "megatron_infer":
        from roll.distributed.strategy.megatron_strategy import MegatronInferStrategy as strategy_cls
    else:
        raise ValueError(f"Unsupported strategy: {name}")
    if sync_wrapper:
        strategy_cls = create_sync_class(strategy_cls)
    return strategy_cls(worker)
