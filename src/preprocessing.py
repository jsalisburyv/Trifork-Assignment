import os
import yaml
from sklearn.model_selection import train_test_split
from annotation_converter import YoloAnnotation
from typing import List, Dict, Tuple
from data_loader import CocoCategory


def split_dataset(
    images: Dict[int, List[str]],
    annotations: Dict[int, List[YoloAnnotation]],
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 33,
) -> Tuple[
    List[Tuple[int, List[str]]],
    List[Tuple[int, List[str]]],
    List[Tuple[int, List[str]]],
    List[Tuple[int, List[YoloAnnotation]]],
    List[Tuple[int, List[YoloAnnotation]]],
    List[Tuple[int, List[YoloAnnotation]]],
]:
    """
    Split a dataset into training, testing, and validation sets.

    Parameters:
    - images (Dict[int, List[str]]): Dictionary of image IDs mapped to their paths.
    - annotations (Dict[int, List[YoloAnnotation]]): Dictionary of image IDs mapped to YOLO annotations.
    - val_size (float): The proportion of the dataset to include in the validation split. Default is 0.15.
    - test_size (float): The proportion of the dataset to include in the test split. Default is 0.15.
    - random_state (int): Seed for the random number generator. Default is 33.

    Returns:
    - Tuple containing lists of training, testing, and validation images and annotations.
    """

    try:
        images_list = [(k, v) for k, v in images.items()]
        annotations_list = [(k, v) for k, v in annotations.items()]

        # Split the dataset into training and temporary sets
        (
            train_images,
            temp_images,
            train_annotations,
            temp_annotations,
        ) = train_test_split(
            images_list,
            annotations_list,
            test_size=(val_size + test_size),
            random_state=random_state,
        )

        # Split the temporary set into test and validation sets
        val_images, test_images, val_annotations, test_annotations = train_test_split(
            temp_images,
            temp_annotations,
            test_size=test_size / (val_size + test_size),
            random_state=random_state,
        )

        return (
            train_images,
            val_images,
            test_images,
            train_annotations,
            val_annotations,
            test_annotations
        )
    except Exception as e:
        raise RuntimeError(f"Error splitting dataset: {e}")


def save_dataset(
    train_images: List[Tuple[int, List[str]]],
    val_images: List[Tuple[int, List[str]]],
    test_images: List[Tuple[int, List[str]]],
    train_annotations: List[Tuple[int, List[YoloAnnotation]]],
    val_annotations: List[Tuple[int, List[YoloAnnotation]]],
    test_annotations: List[Tuple[int, List[YoloAnnotation]]],
    output_path: str,
    coco_categories: List[CocoCategory]
) -> None:
    """
    Save the training, validation, and testing datasets, and create YOLO YAML configuration file.

    Parameters:
    - train_images (List[Tuple[int, List[str]]]): List of training images with IDs and paths.
    - val_images (List[Tuple[int, List[str]]]): List of validation images with IDs and paths.
    - test_images (List[Tuple[int, List[str]]]): List of testing images with IDs and paths.
    - train_annotations (List[Tuple[int, List[YoloAnnotation]]]): List of training annotations with image IDs.
    - val_annotations (List[Tuple[int, List[YoloAnnotation]]]): List of validation annotations with image IDs.
    - test_annotations (List[Tuple[int, List[YoloAnnotation]]]): List of testing annotations with image IDs.
    - output_path (str): Path to the output folder.
    - coco_categories (List[CocoCategory]): List of CocoCategory instances representing class information.

    Raises:
    - RuntimeError: If there is an error during the dataset saving process or YAML file writing process.
    """
    try:
        os.makedirs(output_path, exist_ok=True)
        dir_names = ["train", "validation", "test"]
        __create_folder_and_save_data(
            output_path, dir_names[0], train_images, train_annotations
        )
        __create_folder_and_save_data(
            output_path, dir_names[1], val_images, val_annotations
        )
        __create_folder_and_save_data(
            output_path, dir_names[2], test_images, test_annotations
        )

        _write_yolo_yaml(output_path, *dir_names, coco_categories)
    except Exception as e:
        raise RuntimeError(f"Error saving dataset: {e}")


def __create_folder_and_save_data(
    folder_path: str,
    folder_name: str,
    images: List[Tuple[int, List[str]]],
    annotations: List[Tuple[int, List[YoloAnnotation]]],
) -> None:
    """
    Create a folder and save data (images and annotations) in it.

    Parameters:
    - folder_path (str): Path to the folder.
    - folder_name (str): Name of the folder.
    - images (List[Tuple[int, List[str]]]): List of images with IDs and paths.
    - annotations (List[Tuple[int, List[YoloAnnotation]]]): List of annotations with image IDs.
    """
    try:
        images_path = os.path.join(folder_path, 'images', folder_name)
        annotations_path = os.path.join(folder_path, 'labels', folder_name)
        os.makedirs(images_path, exist_ok=True)
        os.makedirs(annotations_path, exist_ok=True)
        for id, image in images:
            image_path = os.path.join(images_path, str(id) + ".jpg")
            image.save(image_path)
            for anottation_id, anotation_data in annotations:
                if str(anottation_id).zfill(3) == str(id):
                    anotation_path = os.path.join(
                        annotations_path, str(id) + ".txt")
                    YoloAnnotation.write_to_file(
                        anotation_data, anotation_path)
    except Exception as e:
        raise RuntimeError(f"Error creating folder and saving data: {e}")


def _write_yolo_yaml(
    folder_path: str,
    train_rel_path: str,
    val_rel_path: str,
    test_rel_path: str,
    coco_categories: List[CocoCategory],
    filename: str = "yolo",
) -> None:
    """
    Write YOLO YAML file with the specified format.

    Parameters:
    - folder_path (str): Root directory of the dataset.
    - train_rel_path (str): Path to the directory containing training images (relative to 'folder_path').
    - val_rel_path (str): Path to the directory containing validation images (relative to 'folder_path').
    - test_rel_path (str): Path to the directory containing test images (relative to 'folder_path').
    - coco_categories (List[CocoCategory]): List of CocoCategory instances representing class information.
    - filename (str): Name of the YAML file (default is "yolo").

    Raises:
    - RuntimeError: If there is an error during the YAML file writing process.
    """

    try:
        # If different id's are needed should use the same map as in coco_to_yolo()
        category_dict = {category.id -
                         1: category.name for category in coco_categories}

        data = {
            "path": folder_path,
            "train": 'images/' + train_rel_path,
            "val": 'images/' + val_rel_path,
            "test": 'images/' + test_rel_path,
            "names": category_dict,
        }

        file_path = os.path.join(folder_path, f"{filename}.yaml")
        with open(file_path, "w") as yaml_file:
            yaml.dump(data, yaml_file,
                      default_flow_style=False, sort_keys=False)

    except Exception as e:
        raise RuntimeError(f"Error writing YOLO YAML file: {e}")
