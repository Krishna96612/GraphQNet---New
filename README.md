# Requirements & Installation

pip install ultralytics opencv-python opencv-python-headless torch torchvision polars matplotlib torch_geometric

# How to Run GraphQNet

import sys
import os
from ultralytics import YOLO

# 1️⃣ Setup the Path (Use './' for the current folder)
CUSTOM_MODULES_DIR = './' 
if CUSTOM_MODULES_DIR not in sys.path:
    sys.path.insert(0, CUSTOM_MODULES_DIR)

# 2️⃣ Import YOLO and your custom blocks
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils import DEFAULT_CFG
from yolo_custom_modules import MultiHopGNNBlock, QuantumInspiredBlock

# 3️⃣ Inject into YOLO's global namespace
import ultralytics.nn.tasks as yolo_tasks
yolo_tasks.__dict__["MultiHopGNNBlock"] = MultiHopGNNBlock
yolo_tasks.__dict__["QuantumInspiredBlock"] = QuantumInspiredBlock

# 4️⃣ Load your specific YAML architecture
# Ensure this path points to where you extracted the library
model = YOLO("./ultralyticsyolov8_FLIES/ultralytics/cfg/models/v8/yolov8s_gnn_2.yaml")

# 5️⃣ Train the model
results = model.train(
    data="./dog-pose.yaml", # Ensure your .yaml 'path' is also updated
    epochs=600,
    imgsz=640,
    patience=0
)

## ⚠️ Important Things to Note:

1. **YAML Configuration**: You must manually change the `data` and `annotation` paths inside the `.yaml` files of the respective datasets to match your local folder structure.
2. **Model & Dataset Paths**: Ensure the paths provided in the training script (e.g., `model = YOLO(...)`) point correctly to your local unzipped files.
3. **Release Assets**: The modified `ultralytics` folder and the pose datasets are provided as `.zip` files in the **Releases** section. You **must** unzip these into your project root for the code to function.
