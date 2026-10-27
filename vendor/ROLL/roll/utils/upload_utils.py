import errno
import os
import shutil
import stat
import tempfile

from roll.utils.logging import get_logger


logger = get_logger()

uploader_registry = {}

_COPY_TRANSFER_MODE = "copy"
_HARDLINK_TRANSFER_MODE = "hardlink"
_SUPPORTED_TRANSFER_MODES = {_COPY_TRANSFER_MODE, _HARDLINK_TRANSFER_MODE}
_HARDLINK_FALLBACK_ERRNOS = {
    errno.EXDEV,
    errno.EMLINK,
    errno.ENOSYS,
    errno.EPERM,
    getattr(errno, "ENOTSUP", errno.EINVAL),
    getattr(errno, "EOPNOTSUPP", errno.EINVAL),
}


def _atomic_copy(src: str, dst: str) -> str:
    fd, staged_path = tempfile.mkstemp(prefix=".roll-upload-", dir=os.path.dirname(dst))
    os.close(fd)
    try:
        shutil.copy2(src, staged_path)
        os.replace(staged_path, dst)
    finally:
        if os.path.lexists(staged_path):
            os.unlink(staged_path)
    return dst


def _hardlink_or_copy(src: str, dst: str) -> str:
    source_mode = os.stat(src, follow_symlinks=False).st_mode
    if stat.S_ISLNK(source_mode) or not stat.S_ISREG(source_mode):
        return _atomic_copy(src, dst)

    fd, staged_path = tempfile.mkstemp(prefix=".roll-upload-", dir=os.path.dirname(dst))
    os.close(fd)
    os.unlink(staged_path)
    try:
        try:
            os.link(src, staged_path)
        except NotImplementedError:
            shutil.copy2(src, staged_path)
        except OSError as exc:
            if exc.errno not in _HARDLINK_FALLBACK_ERRNOS:
                raise
            logger.warning(f"hard link unavailable for {src}, falling back to copy: {exc}")
            shutil.copy2(src, staged_path)
        os.replace(staged_path, dst)
    finally:
        if os.path.lexists(staged_path):
            os.unlink(staged_path)
    return dst


class FileSystemUploader:
    """
    将本地的ckpt目录上传到文件系统, oss/cpfs在多Role的场景下，
    每个Role会把自己的ckpt dir的内容上传到OUTPUT_DIR/ckpt_id/下
    {
        "type": "file_system",
        "output_dir": /data/oss_bucket_0/llm/models
    }
    """

    def __init__(self, output_dir, transfer_mode=_COPY_TRANSFER_MODE, *args, **kwargs):
        if transfer_mode not in _SUPPORTED_TRANSFER_MODES:
            supported_modes = ", ".join(sorted(_SUPPORTED_TRANSFER_MODES))
            raise ValueError(f"Unknown file-system transfer mode {transfer_mode!r}; expected one of: {supported_modes}")
        self.output_dir = output_dir
        self.transfer_mode = transfer_mode
        logger.info(f"use FileSystemUploader to upload {output_dir} with transfer_mode={transfer_mode}")

    def upload(self, ckpt_id: str, local_state_path: str, **kwargs):
        ckpt_id_output_dir = os.path.join(self.output_dir, ckpt_id)
        os.makedirs(ckpt_id_output_dir, exist_ok=True)
        logger.info(f"{local_state_path} save to {ckpt_id_output_dir}, wait...")
        copy_function = shutil.copy2
        if self.transfer_mode == _HARDLINK_TRANSFER_MODE:
            source_device = os.stat(local_state_path).st_dev
            destination_device = os.stat(ckpt_id_output_dir).st_dev
            if source_device == destination_device:
                copy_function = _hardlink_or_copy
            else:
                logger.warning(
                    f"hard link unavailable across file systems for {local_state_path} and {ckpt_id_output_dir}; "
                    "falling back to copy"
                )
        shutil.copytree(
            local_state_path,
            ckpt_id_output_dir,
            dirs_exist_ok=True,
            copy_function=copy_function,
        )
        logger.info(f"{local_state_path} save to {ckpt_id_output_dir}, done...")


uploader_registry['file_system'] = FileSystemUploader
