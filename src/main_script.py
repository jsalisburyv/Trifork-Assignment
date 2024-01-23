import argparse
import data_loader
import annotation_converter


def main(args):
    coco_images, coco_annotations, coco_categories = data_loader.load_coco_annotations(args.coco_file)
    yolo_annotations = annotation_converter.coco_to_yolo(coco_images, coco_annotations, coco_categories)
    images = list(data_loader.load_images_from_path(args.images_path))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset Preprocessing')
    parser.add_argument('-c', '--coco_file', default='data/coco.json', help='Path to the COCO annotations JSON file.', type=str)
    parser.add_argument('-i', '--images_path', default='data/images', help='Path to the folder containing images.', type=str)
    parser.add_argument('-o', '--output_path', default='output', help='Path to the output folder.', type=str)

    args = parser.parse_args()
    main(args)