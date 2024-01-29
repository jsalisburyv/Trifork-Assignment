import argparse
import data_loader
import annotation_converter
import preprocessing
from PIL import Image
import os


def main(args) -> None:
    coco_data = data_loader.load_coco_annotations(args.coco_file)
    yolo_annotations = annotation_converter.coco_to_yolo(*coco_data)

    sorted_yolo_annotations = dict(sorted(yolo_annotations.items()))

    images = dict(data_loader.load_images_from_path(args.images_path))
    resized_images = {}
    for _id, image in images.items():
        resized_images[_id] = image.resize(args.new_size, Image.NEAREST)
    splitted_dataset = preprocessing.split_dataset(
        resized_images, sorted_yolo_annotations
    )
    preprocessing.save_dataset(*splitted_dataset, args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Preprocessing")
    parser.add_argument(
        "-c",
        "--coco_file",
        required=True,
        help="Path to the COCO annotations JSON file.",
        type=str,
    )
    parser.add_argument(
        "-i",
        "--images_path",
        required=True,
        help="Path to the folder containing images.",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output_path",
        required=True,
        help="Path to the output folder.",
        type=str,
    )
    parser.add_argument(
        "-n", "--new_size", default=(1024, 768), help="New Image size.", type=tuple
    )

    args = parser.parse_args()
    main(args)
