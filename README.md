

```markdown
# 📦 Requirements & Installation
Run this command in your terminal to grab the necessary libraries:

```bash
pip install ultralytics opencv-python opencv-python-headless torch torchvision polars matplotlib torch_geometric
```

# 🚀 How to Run GraphQNet
Follow these steps to initialize the custom architecture and start training.

### 1️⃣ Setup the Path
We'll start by pointing the system to your current folder:

```python
import sys
import os
from ultralytics import YOLO

CUSTOM_MODULES_DIR = './' 
if CUSTOM_MODULES_DIR not in sys.path:
    sys.path.insert(0, CUSTOM_MODULES_DIR)
```

### 2️⃣ Inject Custom Blocks
Now, we import the Multi-Hop GNN and Quantum blocks and "inject" them:

```python
from ultralytics.nn.tasks import PoseModel
import ultralytics.nn.tasks as yolo_tasks
from yolo_custom_modules import MultiHopGNNBlock, QuantumInspiredBlock

# This maps the names in your YAML to the actual code blocks
yolo_tasks.__dict__["MultiHopGNNBlock"] = MultiHopGNNBlock
yolo_tasks.__dict__["QuantumInspiredBlock"] = QuantumInspiredBlock
```

### 3️⃣ Load & Train
Finally, point the script to your unzipped architecture file:

```python
# Load the specific GNN architecture
model = YOLO("./ultralyticsyolov8_FLIES/ultralytics/cfg/models/v8/yolov8s_gnn_2.yaml")

# Begin training
results = model.train(
    data="./dog-pose.yaml", 
    epochs=600,
    imgsz=640,
    patience=0
)
```

### ⚠️ Important Things to Note
* **YAML Config:** You must manually edit the `data` and `annotation` paths inside your `.yaml` files.
* **Unzip First:** Both the modified `ultralytics` folder and the datasets are in the **Releases**. Extract them into your project root.
