import json
import os
from pathlib import Path
from typing import List, Iterable, Tuple
from dataclasses import dataclass
from PIL import Image


@dataclass
class CocoImage:
    """
    Represents an image in COCO format.

    Attributes:
    - id (int): Image identifier.
    - file_name (str): Name of the image file.
    - height (int): Height of the image.
    - width (int): Width of the image.
    """

    id: int
    file_name: str
    height: int
    width: int


@dataclass
class CocoAnnotation:
    """
    Represents an annotation in COCO format.

    Attributes:
    - id (int): Annotation identifier.
    - image_id (int): Identifier of the associated image.
    - category_id (int): Identifier of the category.
    - bbox (List[int]): Bounding box coordinates.
    - area (float): Area of the annotation.
    - iscrowd (int): Indicator for crowd annotations.
    """

    id: int
    image_id: int
    category_id: int
    bbox: List[int]
    area: float
    iscrowd: int


@dataclass
class CocoCategory:
    """
    Represents a category in COCO format.

    Attributes:
    - id (int): Category identifier.
    - name (str): Category name.
    - supercategory (str): Supercategory name.
    """

    id: int
    name: str
    supercategory: str


def load_coco_annotations(
    coco_annotations_path: str,
) -> Tuple[List[CocoImage], List[CocoAnnotation], List[CocoCategory]]:
    """
    Load COCO annotations from a JSON file.

    Parameters:
    - coco_annotations_path (str): Path to the COCO annotations JSON file.

    Returns:
    - Tuple[List[CocoImage], List[CocoAnnotation], List[CocoCategory]]:
      A tuple containing lists of CocoImage, CocoAnnotation, and CocoCategory instances.

    Raises:
    - FileNotFoundError: If the specified file path is not found.
    - ValueError: If there is an issue with the JSON format in the file.
    """

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


def load_images_from_path(images_path: str) -> Iterable[Tuple[str, Image.Image]]:
    """
    Load images from a directory.

    Parameters:
    - images_path (str): Path to the folder containing images.

    Returns:
    - Iterable[Tuple[str, Image.Image]]: Iterator yielding tuples of image ID and corresponding Image instance.

    Raises:
    - FileNotFoundError: If the specified directory path is not found.
    - RuntimeError: If there is an error while loading images.
    """
    try:
        for filename in os.listdir(images_path):
            if filename.endswith(".jpg"):
                path = Path(images_path, filename)
                yield path.stem, Image.open(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Directory not found: {images_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading images from path: {e}")
