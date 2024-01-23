from typing import List
from dataclasses import dataclass
from data_loader import CocoAnnotation, CocoImage, CocoCategory

@dataclass
class YoloAnnotation:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

def coco_to_yolo(coco_images: List[CocoImage], coco_annotations: List[CocoAnnotation], coco_categories: List[CocoCategory]) -> List[YoloAnnotation]:
    """
    Convert COCO annotations to YOLO format.

    Parameters:
    - coco_annotations (List[CocoAnnotation]): List of CocoAnnotation instances.
    - coco_images (List[CocoImage]): List of CocoImage instances.

    Returns:
    - List[YoloAnnotation]: List of YoloAnnotation instances.
    """
    yolo_annotations = {}
    # Used
    class_mapping = {category.id: category.id - 1 for category in coco_categories}
    
    for img in coco_images:
        yolo_annotations[img.id] = []
        for annotation in coco_annotations:
            if annotation.image_id == img.id:
                # Convert COCO bounding box coordinates to YOLO format
                x_center = (annotation.bbox[0] + annotation.bbox[2]) / (2 * img.width)
                y_center = (annotation.bbox[1] + annotation.bbox[3]) / (2 * img.height)
                width = (annotation.bbox[2] - annotation.bbox[0]) / img.width
                height = (annotation.bbox[3] - annotation.bbox[1]) / img.height

                # Map COCO category_id to YOLO class_id
                class_id = class_mapping.get(annotation.category_id, -1)

                # Create YoloAnnotation instance
                yolo_annotation = YoloAnnotation(class_id, x_center, y_center, width, height)
                yolo_annotations[img.id].append(yolo_annotation)

    return yolo_annotations
