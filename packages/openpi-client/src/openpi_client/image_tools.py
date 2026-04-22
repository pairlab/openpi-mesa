import numpy as np
from PIL import Image


def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """Converts an image to uint8 if it is a float image.

    This is important for reducing the size of the image when sending it over the network.
    """
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img


def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for multiple images using PIL. Resizes a batch of images to a target height.

    Args:
        images: A batch of images in [..., height, width, channel] format.
        height: The target height of the image.
        width: The target width of the image.
        method: The interpolation method to use. Default is bilinear.

    Returns:
        The resized images in [..., height, width, channel].
    """
    # If the images are already the correct size, return them as is.
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape

    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(Image.fromarray(im), height, width, method=method) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> Image.Image:
    """Replicates tf.image.resize_with_pad for one image using PIL. Resizes an image to a target height and
    width without distortion by padding with zeros.

    Unlike the jax version, note that PIL uses [width, height, channel] ordering instead of [batch, h, w, c].
    """
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image  # No need to resize if the image is already the correct size.

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image


def resize_with_pad_depth(images: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize float32 depth images with aspect-preserving pad, mirroring ``resize_with_pad``.

    Uses scipy-free nearest-neighbour sampling on the raw float32 data so metric depth values
    are preserved (PIL's uint8 path would lose precision). Input shape ``(..., H, W, 1)``;
    output shape ``(..., height, width, 1)``.
    """
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape
    flat = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_depth_nearest(im, height, width) for im in flat])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_depth_nearest(depth: np.ndarray, height: int, width: int) -> np.ndarray:
    """Nearest-neighbour resize + centred zero pad for a single ``(H, W, 1)`` depth image."""
    depth_2d = depth.squeeze(axis=-1).astype(np.float32, copy=False)
    cur_h, cur_w = depth_2d.shape
    if (cur_h, cur_w) == (height, width):
        return depth_2d[..., None]

    ratio = max(cur_w / width, cur_h / height)
    resized_h = int(cur_h / ratio)
    resized_w = int(cur_w / ratio)

    # Nearest-neighbour via integer index arithmetic (preserves exact depth values).
    row_idx = (np.arange(resized_h) * (cur_h / resized_h)).astype(np.int64).clip(0, cur_h - 1)
    col_idx = (np.arange(resized_w) * (cur_w / resized_w)).astype(np.int64).clip(0, cur_w - 1)
    resized = depth_2d[row_idx[:, None], col_idx[None, :]]

    padded = np.zeros((height, width), dtype=np.float32)
    pad_h = max(0, (height - resized_h) // 2)
    pad_w = max(0, (width - resized_w) // 2)
    padded[pad_h : pad_h + resized_h, pad_w : pad_w + resized_w] = resized
    return padded[..., None]
