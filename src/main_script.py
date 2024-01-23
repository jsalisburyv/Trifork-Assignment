import argparse
import json
import os


def load_coco_annotations(coco_annotations_file):
    with open(os.path.normpath(coco_annotations_file), 'r') as file:
        coco_annotations = json.load(file)
    return coco_annotations
    
def load_images_from_folder(images_path):
    image_filenames = [filename for filename in os.listdir(images_path) if filename.endswith('.jpg')]
    return image_filenames

def main(args):
    coco_annotations = load_coco_annotations(args.annotations_file)
    print(coco_annotations)
    image_filenames = load_images_from_folder(args.images_path)
    print(image_filenames)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset Preprocessing')
    parser.add_argument('-a', '--annotations_file', default='data/coco.json', help='Path to the COCO annotations JSON file.', type=str)
    parser.add_argument('-i', '--images_path', default='data/images', help='Path to the folder containing images.', type=str)
    parser.add_argument('-o', '--output_path', default='output', help='Path to the output folder.', type=str)

    args = parser.parse_args()
    main(args)