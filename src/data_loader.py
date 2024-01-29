import json
import os
from pathlib import Path
from typing import List, Iterable, Tuple
from dataclasses import dataclass
from PIL import Image


@dataclass
class CocoImage:
    id: int
    file_name: str
    height: int
    width: int


@dataclass
class CocoAnnotation:
    id: int
    image_id: int
    category_id: int
    bbox: List[int]
    area: float
    iscrowd: int


@dataclass
class CocoCategory:
    id: int
    name: str
    supercategory: str


def load_coco_annotations(
    coco_annotations_path: str,
) -> Tuple[List[CocoImage], List[CocoAnnotation], List[CocoCategory]]:
    try:
        with open(coco_annotations_path, "r") as file:
            coco_data = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {coco_annotations_path}")
    except json.JSONDecodeError:
        raise ValueError(
            f"Invalid JSON format in file: {coco_annotations_path}")

    coco_images = [
        CocoImage(**image_data) for image_data in coco_data.get("images", [])
    ]
    coco_annotations = [
        CocoAnnotation(**annotation_data)
        for annotation_data in coco_data.get("annotations", [])
    ]
    coco_categories = [
        CocoCategory(**category_data)
        for category_data in coco_data.get("categories", [])
    ]

    return coco_images, coco_annotations, coco_categories


# def load_images_from_path(images_path: str) -> Iterable[Tuple[str, Image.Image]]:
#     try:
#         for filename in os.listdir(images_path):
#             if filename.endswith(".jpg"):
#                 path = Path(images_path, filename)
#                 with Image.open(path) as image:
#                     yield path.stem, image
#     except FileNotFoundError:
#         raise FileNotFoundError(f"Directory not found: {images_path}")
#     except Exception as e:
#         raise RuntimeError(f"Error loading images from path: {e}")
def load_images_from_path(images_path: str) -> Iterable[Tuple[str, Image.Image]]:
    try:
        for filename in os.listdir(images_path):
            if filename.endswith(".jpg"):
                path = Path(images_path, filename)
                yield path.stem, Image.open(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Directory not found: {images_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading images from path: {e}")
