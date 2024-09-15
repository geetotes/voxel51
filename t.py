import fiftyone as fo
import fiftyone.zoo as foz
from ultralytics import YOLO


# dataset = foz.load_zoo_dataset("quickstart")

name = "my-dataset"

labels_path = "./COD10K-v3/Train/CAM_Instance_Train.json"
data_path = "./COD10K-v3/Train/Image"

# Create the dataset
trained_dataset = fo.Dataset.from_dir(
    labels_path=labels_path,
    data_path=data_path,
    dataset_type=fo.types.COCODetectionDataset,
    name="COD10K",
    overwrite=True,
)

# YOLO format requires a common classes list
classes = trained_dataset.default_classes

trained_dataset.export(
    export_dir="./tmp/yolo",
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="detections",
    split="train",
    classes=classes,
)

labels_path = "./COD10K-v3/Test/CAM_Instance_Test.json"
data_path = "./COD10K-v3/Test/Image"


untrained_dataset = fo.Dataset.from_dir(
    labels_path=labels_path,
    data_path=data_path,
    dataset_type=fo.types.COCODetectionDataset,
    name="untrained-COD10K",
    overwrite=True,
)

untrained_dataset.export(
    export_dir="./tmp/yolo",
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="detections",
    split="val",
    classes=classes,
)

trained_dataset.tag_samples("trained")
untrained_dataset.tag_samples("testin")
trained_dataset.merge_samples(untrained_dataset)

working_dataset = trained_dataset[0:100]

model = YOLO("yolov8n-cls.pt")
working_dataset.apply_model(model, label_field="classif")

session = fo.launch_app(working_dataset)

# View summary info about the dataset
print(working_dataset)

session.wait()

