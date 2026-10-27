"""Runtime compatibility patches for the EnvScaler RL job."""

import importlib
import importlib.abc
import importlib.machinery
import sys


_MCORE_UTILS_MODULE = "mcore_adapter.utils"
_MCORE_PATCH_MARKER = "_envscaler_fast_megatron_llama_patch"
_RAY_WORKER_MODULE = "ray._private.worker"
_RAY_ACCELERATOR_IDS_PATCH_MARKER = "_envscaler_single_visible_accelerator_patch"
_TE_DPA_UTILS_MODULE = "transformer_engine.pytorch.attention.dot_product_attention.utils"
_TE_FLASH_ATTN_PATCH_MARKER = "_envscaler_te_flash_attn_version_patch"
_TARGET_MODULE = "megatron.core.transformer.utils"
_PATCH_MARKER = "_envscaler_bool_attention_mask_patch"


def _patch_mcore_adapter_utils(module):
    if getattr(module, _MCORE_PATCH_MARKER, False):
        return

    def is_megatron_llama():
        if not hasattr(is_megatron_llama, "cached_value"):
            from importlib import metadata

            try:
                metadata.distribution("megatron-llama-core")
                is_megatron_llama.cached_value = True
            except Exception:
                is_megatron_llama.cached_value = False
        return is_megatron_llama.cached_value

    module.is_megatron_llama = is_megatron_llama
    setattr(module, _MCORE_PATCH_MARKER, True)


class _MCoreUtilsPatchLoader(importlib.abc.Loader):
    def __init__(self, wrapped_loader):
        self._wrapped_loader = wrapped_loader

    def create_module(self, spec):
        create_module = getattr(self._wrapped_loader, "create_module", None)
        if create_module is None:
            return None
        return create_module(spec)

    def exec_module(self, module):
        self._wrapped_loader.exec_module(module)
        _patch_mcore_adapter_utils(module)


class _MCoreUtilsPatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != _MCORE_UTILS_MODULE:
            return None

        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _MCoreUtilsPatchLoader(spec.loader)
        return spec


def _patch_ray_worker_accelerator_ids(module):
    worker_cls = module.Worker
    if getattr(worker_cls, _RAY_ACCELERATOR_IDS_PATCH_MARKER, False):
        return

    original_get_ids = worker_cls.get_accelerator_ids_for_accelerator_resource

    def get_accelerator_ids_for_accelerator_resource(self, resource_name, resource_regex):
        try:
            return original_get_ids(self, resource_name, resource_regex)
        except IndexError:
            original_ids = self.original_visible_accelerator_ids.get(resource_name)
            if original_ids is not None and len(original_ids) == 1:
                return list(original_ids)
            raise

    get_accelerator_ids_for_accelerator_resource.__name__ = original_get_ids.__name__
    get_accelerator_ids_for_accelerator_resource.__doc__ = original_get_ids.__doc__
    worker_cls.get_accelerator_ids_for_accelerator_resource = get_accelerator_ids_for_accelerator_resource
    setattr(worker_cls, _RAY_ACCELERATOR_IDS_PATCH_MARKER, True)


class _RayWorkerPatchLoader(importlib.abc.Loader):
    def __init__(self, wrapped_loader):
        self._wrapped_loader = wrapped_loader

    def create_module(self, spec):
        create_module = getattr(self._wrapped_loader, "create_module", None)
        if create_module is None:
            return None
        return create_module(spec)

    def exec_module(self, module):
        self._wrapped_loader.exec_module(module)
        _patch_ray_worker_accelerator_ids(module)


class _RayWorkerPatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != _RAY_WORKER_MODULE:
            return None

        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _RayWorkerPatchLoader(spec.loader)
        return spec


def _patch_te_flash_attn_max_version(module):
    if getattr(module, _TE_FLASH_ATTN_PATCH_MARKER, False):
        return

    try:
        from importlib import metadata
        from packaging.version import Version

        installed_version = Version(metadata.version("flash-attn"))
        tested_upper_bound = Version("2.8.3.post1")
        current_max_version = module.FlashAttentionUtils.max_version
    except Exception:
        setattr(module, _TE_FLASH_ATTN_PATCH_MARKER, True)
        return

    if current_max_version < installed_version <= tested_upper_bound:
        module.FlashAttentionUtils.max_version = installed_version

    setattr(module, _TE_FLASH_ATTN_PATCH_MARKER, True)


class _TEFlashAttnPatchLoader(importlib.abc.Loader):
    def __init__(self, wrapped_loader):
        self._wrapped_loader = wrapped_loader

    def create_module(self, spec):
        create_module = getattr(self._wrapped_loader, "create_module", None)
        if create_module is None:
            return None
        return create_module(spec)

    def exec_module(self, module):
        self._wrapped_loader.exec_module(module)
        _patch_te_flash_attn_max_version(module)


class _TEFlashAttnPatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != _TE_DPA_UTILS_MODULE:
            return None

        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _TEFlashAttnPatchLoader(spec.loader)
        return spec


def _patch_megatron_transformer_utils(module):
    if getattr(module, _PATCH_MARKER, False):
        return

    original_attention_mask_func = module.attention_mask_func

    def attention_mask_func(attention_scores, attention_mask):
        if attention_mask is not None and getattr(attention_mask, "dtype", None) is not None:
            import torch

            if attention_mask.dtype is not torch.bool:
                attention_mask = attention_mask.bool()
        return original_attention_mask_func(attention_scores, attention_mask)

    attention_mask_func.__name__ = original_attention_mask_func.__name__
    attention_mask_func.__doc__ = original_attention_mask_func.__doc__
    module.attention_mask_func = attention_mask_func
    setattr(module, _PATCH_MARKER, True)

    dot_product_attention = sys.modules.get("megatron.core.transformer.dot_product_attention")
    if dot_product_attention is not None:
        dot_product_attention.attention_mask_func = attention_mask_func


class _MegatronMaskPatchLoader(importlib.abc.Loader):
    def __init__(self, wrapped_loader):
        self._wrapped_loader = wrapped_loader

    def create_module(self, spec):
        create_module = getattr(self._wrapped_loader, "create_module", None)
        if create_module is None:
            return None
        return create_module(spec)

    def exec_module(self, module):
        self._wrapped_loader.exec_module(module)
        _patch_megatron_transformer_utils(module)


class _MegatronMaskPatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET_MODULE:
            return None

        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _MegatronMaskPatchLoader(spec.loader)
        return spec


if _TARGET_MODULE in sys.modules:
    _patch_megatron_transformer_utils(sys.modules[_TARGET_MODULE])
elif not any(isinstance(finder, _MegatronMaskPatchFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _MegatronMaskPatchFinder())

if _MCORE_UTILS_MODULE in sys.modules:
    _patch_mcore_adapter_utils(sys.modules[_MCORE_UTILS_MODULE])
elif not any(isinstance(finder, _MCoreUtilsPatchFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _MCoreUtilsPatchFinder())

if _RAY_WORKER_MODULE in sys.modules:
    _patch_ray_worker_accelerator_ids(sys.modules[_RAY_WORKER_MODULE])
elif not any(isinstance(finder, _RayWorkerPatchFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _RayWorkerPatchFinder())

if _TE_DPA_UTILS_MODULE in sys.modules:
    _patch_te_flash_attn_max_version(sys.modules[_TE_DPA_UTILS_MODULE])
elif not any(isinstance(finder, _TEFlashAttnPatchFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _TEFlashAttnPatchFinder())

try:
    from importlib.metadata import version

    from packaging.version import Version

    torch_version = Version(version("torch"))
    # Torch 2.6's inductor config references custom_graph_pass while Dynamo is
    # still being imported. Finish Dynamo first to avoid a circular/partial
    # module, then preload the attribute used by both Torch 2.6 and 2.8.
    if torch_version < Version("2.8"):
        import torch._dynamo  # noqa: F401
    import torch._inductor.custom_graph_pass  # noqa: F401
except Exception:
    pass
