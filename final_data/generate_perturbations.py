"""
Generate 6 selected perturbations with first 3 severity levels
For ocr1.0 and ocr2.0 datasets
"""
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
from scipy.ndimage import gaussian_filter, map_coordinates
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# 6 perturbations with 3 severity levels
PERTURBATIONS = {
    "pixelate": [1.5, 2, 2.5],
    "glass_blur": [2, 2.5, 3],
    "color_shift": [3, 4, 5],
    "elastic": [10, 15, 20],
    "motion_blur": [5, 6, 7],
    "snow": [0.1, 0.2, 0.3],
}

# PERTURBATIONS = {
    # "pixelate": [1,3],
    # "glass_blur": [1.6,1.7],  # integer pixel radius
    # "color_shift": [5],
    # "elastic": [12,18],
    # "motion_blur": [5.5,6.5],
    # "snow": [0.15],
# }


def add_pixelate(image, block_size):
    block_size = int(max(1, block_size))
    w, h = image.size
    small_w = max(1, w // block_size)
    small_h = max(1, h // block_size)
    small = image.resize((small_w, small_h), Image.NEAREST)
    return small.resize((w, h), Image.NEAREST)

def add_glass_blur(image, radius):
    """Frosted glass blur: displace pixels then smooth displacement with Gaussian"""
    img_array = np.array(image, dtype=np.float32)
    h, w, c = img_array.shape
    sigma = max(0.5, radius * 0.4)  # adaptive sigma: larger radius = smoother
    dx = np.random.randint(-int(radius), int(radius) + 1, size=(h, w))
    dy = np.random.randint(-int(radius), int(radius) + 1, size=(h, w))
    dx = gaussian_filter(dx.astype(np.float32), sigma)
    dy = gaussian_filter(dy.astype(np.float32), sigma)
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    new_x = np.clip(x_coords + dx.astype(int), 0, w - 1)
    new_y = np.clip(y_coords + dy.astype(int), 0, h - 1)
    result = img_array[new_y, new_x, :]
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

def add_color_shift(image, offset):
    """Chromatic aberration: shift R/B channels by offset pixels"""
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    result = img_array.copy()

    # Shift R channel right
    if offset > 0:
        result[:, offset:, 0] = img_array[:, :-offset, 0]
        result[:, :offset, 0] = img_array[:, 0:1, 0]

    # Shift B channel left
    if offset > 0:
        result[:, :-offset, 2] = img_array[:, offset:, 2]
        result[:, -offset:, 2] = img_array[:, -1:, 2]

    return Image.fromarray(result)

def add_elastic(image, alpha):
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    sigma = 4.0  # small sigma = more local, visible distortion
    dx = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    indices = [np.clip(y + dy, 0, h - 1), np.clip(x + dx, 0, w - 1)]
    distorted = np.zeros_like(img_array)
    for i in range(img_array.shape[2]):
        distorted[:, :, i] = map_coordinates(img_array[:, :, i], indices, order=1)
    return Image.fromarray(distorted.astype(np.uint8))

def add_motion_blur(image, kernel_size):
    img_array = np.array(image)
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    blurred = cv2.filter2D(img_array, -1, kernel)
    return Image.fromarray(blurred)

def add_snow(image, intensity):
    img_array = np.array(image).astype(np.float32)
    h, w = img_array.shape[:2]
    snow_mask = (np.random.rand(h, w) < intensity).astype(np.float32)
    img_array = img_array * (1 - snow_mask[:, :, None] * 0.8) + snow_mask[:, :, None] * 255
    return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

FUNC_MAP = {
    "pixelate": add_pixelate,
    "glass_blur": add_glass_blur,
    "color_shift": add_color_shift,
    "elastic": add_elastic,
    "motion_blur": add_motion_blur,
    "snow": add_snow,
}

def process_image(args):
    img_path, dataset, base_dir = args
    try:
        image = Image.open(img_path).convert('RGB')
        for pert_name, levels in PERTURBATIONS.items():
            func = FUNC_MAP[pert_name]
            for level in levels:
                output_dir = base_dir / "perturbations" / f"{pert_name}_{level}" / dataset
                perturbed = func(image, level)
                perturbed.save(output_dir / img_path.name)
        return None
    except Exception as e:
        return f"Error {img_path.name}: {e}"

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    datasets = ["ocr1.0", "ocr2.0"]
    NUM_WORKERS = min(8, multiprocessing.cpu_count())

    # Pre-create all output dirs
    for dataset in datasets:
        for pert_name, levels in PERTURBATIONS.items():
            for level in levels:
                (base_dir / "perturbations" / f"{pert_name}_{level}" / dataset).mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        input_dir = base_dir / dataset / "images"
        img_files = list(input_dir.glob("*.[jp][pn][g]")) + list(input_dir.glob("*.jpeg"))
        print(f"\n{dataset}: {len(img_files)} images, {NUM_WORKERS} workers")

        args_list = [(p, dataset, base_dir) for p in img_files]

        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            results = list(tqdm(executor.map(process_image, args_list), total=len(args_list), desc=f"  {dataset}"))

        errors = [r for r in results if r is not None]
        for err in errors:
            print(f"  {err}")

    print("\nAll perturbations generated!")
