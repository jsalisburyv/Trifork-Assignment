import os
from sklearn.model_selection import train_test_split
from annotation_converter import YoloAnnotation
from typing import List, Dict, Tuple


def split_dataset(
    images: Dict[int, List[str]],
    annotations: Dict[int, List[YoloAnnotation]],
    test_size: float = 0.2,
    val_size: float = 0.1,
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
    ----------
    - image_paths (list): List of image paths.
    - annotations (list): List of corresponding annotations.
    - test_size (float): The proportion of the dataset to include in the test split. Default is 0.2.
    - val_size (float): The proportion of the dataset to include in the validation split. Default is 0.1.
    - random_state (int): Seed for the random number generator. Default is 33.

    Returns:
    - train_paths, test_paths, train_annotations, test_annotations, val_paths, val_annotations
    """
    images_list = [(k, v) for k, v in images.items()]
    annotations_list = [(k, v) for k, v in annotations.items()]

    # Split the dataset into training and temporary sets
    train_images, temp_images, train_annotations, temp_annotations = train_test_split(
        images_list,
        annotations_list,
        test_size=(test_size + val_size),
        random_state=random_state,
    )

    # Split the temporary set into test and validation sets
    test_images, val_images, test_annotations, val_annotations = train_test_split(
        temp_images,
        temp_annotations,
        test_size=val_size / (test_size + val_size),
        random_state=random_state,
    )

    return (
        train_images,
        test_images,
        val_images,
        train_annotations,
        test_annotations,
        val_annotations,
    )


def save_dataset(
    train_images: List[Tuple[int, List[str]]],
    test_images: List[Tuple[int, List[str]]],
    val_images: List[Tuple[int, List[str]]],
    train_annotations: List[Tuple[int, List[YoloAnnotation]]],
    test_annotations: List[Tuple[int, List[YoloAnnotation]]],
    val_annotations: List[Tuple[int, List[YoloAnnotation]]],
    output_path: str,
) -> None:
    os.makedirs(output_path, exist_ok=True)
    # dir_names = ["train", "test", "validation"]
    __create_folder_and_save_data(
        os.path.join(output_path, "train"), train_images, train_annotations
    )
    __create_folder_and_save_data(
        os.path.join(output_path, "test"), test_images, test_annotations
    )
    __create_folder_and_save_data(
        os.path.join(output_path, "validation"), val_images, val_annotations
    )


def __create_folder_and_save_data(
    folder_path: str,
    images: List[Tuple[int, List[str]]],
    annotations: List[Tuple[int, List[YoloAnnotation]]],
) -> None:
    os.makedirs(folder_path, exist_ok=True)
    for id, image in images:
        image_path = os.path.join(folder_path, id + ".jpg")
        image.save(image_path)
        for anottation_id, anotation_data in annotations:
            if str(anottation_id).zfill(3) == id:
                anotation_path = os.path.join(folder_path, id + ".txt")
                YoloAnnotation.write_to_file(anotation_data, anotation_path)
