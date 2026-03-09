import contextlib
import os
import tempfile
from collections.abc import Iterable
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

import wandb

if TYPE_CHECKING:
    from PIL import Image
else:
    try:
        from PIL import Image
    except ImportError:
        Image = None


def _load_image(obj) -> np.ndarray:
    """Convert various image inputs into a numpy uint8 HWC array.

    Accepts: file path (str/Path), bytes, PIL.Image, numpy array.
    """
    # Path or string -> open with PIL
    if isinstance(obj, (str | Path)):
        if Image is None:
            raise RuntimeError("Pillow is required to load image paths")
        img = Image.open(str(obj)).convert("RGB")
        return np.asarray(img, dtype=np.uint8)

    # bytes -> open with PIL
    if isinstance(obj, bytes | bytearray):
        if Image is None:
            raise RuntimeError("Pillow is required to load image bytes")
        img = Image.open(BytesIO(obj)).convert("RGB")
        return np.asarray(img, dtype=np.uint8)

    # PIL Image
    if Image is not None and isinstance(obj, Image.Image):
        return np.asarray(obj.convert("RGB"), dtype=np.uint8)

    # numpy array
    if isinstance(obj, np.ndarray):
        arr = obj
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        # if CHW -> convert to HWC
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        return arr

    raise TypeError(f"Unsupported image type: {type(obj)}")


def upload_neural_renderer_image(image, key: str, step: int | None = None):
    """Logs a single image to wandb.

    The `image` argument may be:
    - a file path (str or Path) to an image file
    - raw bytes of an image (e.g. PNG/JPEG)
    - a PIL Image
    - a numpy array (HWC or CHW)
    """
    arr = _load_image(image)
    wandb.log({key: wandb.Image(arr)}, step=step)


def _frames_to_array_list(frames: Iterable) -> list:
    """Convert an iterable of frame inputs into a list of numpy arrays.

    Each frame may be any type accepted by `_load_image`.
    """
    out = []
    for f in frames:
        out.append(_load_image(f))  # noqa: PERF401
    return out


def upload_neural_renderer_gif(frames, key: str, step: int | None = None, fps: int = 10):
    """Logs a GIF or sequence of frames to wandb.

    key: the wandb key to log under
    step: optional step number to log at (for wandb)
    fps: frames per second for the GIF (default: 10)

    `frames` may be:
    - a filesystem path (str/Path) to an already-created GIF/video file
    - raw bytes of a GIF file
    - an iterable (list/tuple) of frames where each frame is a path/bytes/PIL/numpy
    - a numpy array of shape (T, H, W, C)
    """
    # If a path-like was provided, pass it straight to wandb.Video
    if isinstance(frames, (str | Path)):
        wandb.log({key: wandb.Video(str(frames), fps=fps, format="gif")}, step=step)
        print(f"Logged GIF file {frames} to wandb under key '{key}' at step {step}")
        return

    # If bytes (presumably a GIF file) -> write to temp file and pass path
    # If bytes (presumably a GIF file) -> write to temp file and pass path
    if isinstance(frames, (bytes | bytearray)):
        # 1. Use a context manager to handle the initial file creation/writing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gif") as tf:
            tf.write(frames)
            tf.flush()
            temp_path = tf.name  # Store the path to use outside the 'with' block

        # 2. Now that the file is closed (but not deleted), wandb can safely read it
        try:
            wandb.log({key: wandb.Video(temp_path, fps=fps, format="gif")}, step=step)
            print(f"Logged GIF bytes to wandb under key '{key}' at step {step}")
        finally:
            # 3. Manually clean up the file
            with contextlib.suppress(FileNotFoundError):
                os.remove(temp_path)
        return

    # If numpy array of shape (T,H,W,C)
    if isinstance(frames, np.ndarray):
        if frames.ndim == 4:
            wandb.log({key: wandb.Video(frames, fps=fps, format="gif")}, step=step)
            print(f"Logged GIF numpy array with {frames.shape[0]} frames to wandb under key '{key}' at step {step}")
            return
        else:
            raise TypeError("numpy array input for GIF must be 4D (T,H,W,C)")

    # Otherwise assume iterable of frames
    if isinstance(frames, Iterable):
        frame_list = _frames_to_array_list(frames)
        # Convert list of arrays to a single numpy array (T, H, W, C)
        # This satisfies wandb.Video input requirement
        video_array = np.stack(frame_list)
        wandb.log({key: wandb.Video(video_array, fps=fps, format="gif")}, step=step)
        print(f"Logged GIF with {len(frame_list)} frames to wandb under key '{key}' at step {step}")
        return

    raise TypeError(f"Unsupported frames input type: {type(frames)}")
