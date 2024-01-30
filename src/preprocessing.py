import os
from sklearn.model_selection import train_test_split
from annotation_converter import YoloAnnotation
from typing import List, Dict, Tuple


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
) -> None:
    """
    Save the training, testing, and validation datasets.

    Parameters:
    - train_images (List[Tuple[int, List[str]]]): List of training images with IDs and paths.
    - val_images (List[Tuple[int, List[str]]]): List of validation images with IDs and paths.
    - test_images (List[Tuple[int, List[str]]]): List of testing images with IDs and paths.
    - train_annotations (List[Tuple[int, List[YoloAnnotation]]]): List of training annotations with image IDs.
    - val_annotations (List[Tuple[int, List[YoloAnnotation]]]): List of validation annotations with image IDs.
    - test_annotations (List[Tuple[int, List[YoloAnnotation]]]): List of testing annotations with image IDs.
    - output_path (str): Path to the output folder.
    """
    try:
        os.makedirs(output_path, exist_ok=True)
        # dir_names = ["train", "test", "validation"]
        __create_folder_and_save_data(
            os.path.join(output_path, "train"), train_images, train_annotations
        )
        __create_folder_and_save_data(
            os.path.join(
                output_path, "validation"), val_images, val_annotations
        )
        __create_folder_and_save_data(
            os.path.join(output_path, "test"), test_images, test_annotations
        )
    except Exception as e:
        raise RuntimeError(f"Error saving dataset: {e}")


def __create_folder_and_save_data(
    folder_path: str,
    images: List[Tuple[int, List[str]]],
    annotations: List[Tuple[int, List[YoloAnnotation]]],
) -> None:
    """
    Create a folder and save data (images and annotations) in it.

    Parameters:
    - folder_path (str): Path to the folder.
    - images (List[Tuple[int, List[str]]]): List of images with IDs and paths.
    - annotations (List[Tuple[int, List[YoloAnnotation]]]): List of annotations with image IDs.
    """
    try:
        os.makedirs(folder_path, exist_ok=True)
        for id, image in images:
            image_path = os.path.join(folder_path, str(id) + ".jpg")
            image.save(image_path)
            for anottation_id, anotation_data in annotations:
                if str(anottation_id).zfill(3) == str(id):
                    anotation_path = os.path.join(
                        folder_path, str(id) + ".txt")
                    YoloAnnotation.write_to_file(
                        anotation_data, anotation_path)
    except Exception as e:
        raise RuntimeError(f"Error creating folder and saving data: {e}")
