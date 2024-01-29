from typing import List, Dict
from dataclasses import dataclass
from data_loader import CocoAnnotation, CocoImage, CocoCategory


@dataclass
class YoloAnnotation:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def to_string(self) -> str:
        return f"{self.class_id} {self.x_center} {self.y_center} {self.width} {self.height}"

    @staticmethod
    def write_to_file(annotations: List["YoloAnnotation"], output_path: str) -> None:
        try:
            with open(output_path, "w") as file:
                for annotation in annotations:
                    line = annotation.to_string() + "\n"
                    file.write(line)
        except FileNotFoundError:
            raise FileNotFoundError(f"Output path not found: {output_path}")
        except Exception as e:
            raise RuntimeError(f"Error writing YOLO annotations to file: {e}")


def coco_to_yolo(
    coco_images: List[CocoImage],
    coco_annotations: List[CocoAnnotation],
    coco_categories: List[CocoCategory],
) -> Dict[int, List[YoloAnnotation]]:
    try:
        yolo_annotations = {}
        class_mapping = {category.id: category.id -
                         1 for category in coco_categories}

        for img in coco_images:
            yolo_annotations[img.id] = []
            for annotation in coco_annotations:
                if annotation.image_id == img.id:
                    # Convert COCO bounding box coordinates to YOLO format
                    x_center = (
                        annotation.bbox[0] + annotation.bbox[2]) / (2 * img.width)
                    y_center = (
                        annotation.bbox[1] + annotation.bbox[3]) / (2 * img.height)
                    width = (annotation.bbox[2] -
                             annotation.bbox[0]) / img.width
                    height = (annotation.bbox[3] -
                              annotation.bbox[1]) / img.height

                    # Map COCO category_id to YOLO class_id
                    class_id = class_mapping.get(annotation.category_id, -1)

                    # Create YoloAnnotation instance
                    yolo_annotation = YoloAnnotation(
                        class_id, x_center, y_center, width, height
                    )
                    yolo_annotations[img.id].append(yolo_annotation)

        return yolo_annotations
    except Exception as e:
        raise RuntimeError(
            f"Error converting COCO annotations to YOLO format: {e}")
