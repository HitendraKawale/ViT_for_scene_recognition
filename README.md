Vision Transformers for Scene Recognition: A Comparative Analysis
This project provides a comprehensive framework for training and evaluating deep learning models for multi-class scene recognition. It includes a comparative analysis of three distinct architectures: a standard Vision Transformer (ViT-base), a self-supervised Vision Transformer (DINOv2), and a classic Deep Convolutional Neural Network (ResNet50).

The models are fine-tuned on the Places2_simp dataset, which contains 40,000 images across 40 scene categories. The final evaluation demonstrates the superior performance of the self-supervised DINOv2 model, which achieved a    

top-1 accuracy of 80.82% on a custom-curated test set.


Repository Structure
.
├── configs/              # YAML configuration files for experiments
├── data/         
│   ├── Places2_simp/
│   └── custom_test_set/
├── results/              # All output files (models, logs, figures)
│   └── runs/
├── scripts/              # Main scripts
│   ├── train_from_config.py
│   ├── evaluate.py
│   ├── visualize_predictions.py
│   └── visualize_attention.py
├── utils/                # Helper modules
│   └── dataset.py
├──.gitignore
└── requirements.txt


## Getting Started

### Prerequisites

-   Python 3.10+
-   PyTorch
-   A CUDA-enabled GPU is highly recommended for training.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/HitendraKawale/ViT_for_scene_recognition.git](https://github.com/HitendraKawale/ViT_for_scene_recognition.git)
    cd ViT_for_scene_recognition
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv vit_env
    source vit_env/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the Dataset:**
    Download the `Places2_simp.zip` dataset and unzip it into the `data/` directory.

## Usage

All scripts are run from the root of the project directory.

### Training a Model

To train a model, select a configuration file from the `configs/` directory and run the main training script.

```bash
python scripts/train_from_config.py --config configs/dino_config.yaml
A new folder containing the trained model (vit_best.pth), TensorBoard logs, and a confusion matrix will be created in results/runs/.

Evaluating on the Test Set
To evaluate a trained model on your custom test set, use the evaluate.py script. You must provide the path to the trained model, the test data directory, and the model's architecture name.

Bash

python scripts/evaluate.py \
    --model_path "path/to/your/vit_best.pth" \
    --test_dir "data/custom_test_set/" \
    --model_name "facebook/dinov2-base"
Visualizing Predictions
To visualize the top-5 predictions for a trained model on the validation set:

Bash

# For correctly classified images
python scripts/visualize_predictions.py \
    --model_path "path/to/your/vit_best.pth" \
    --model_name "facebook/dinov2-base" \
    --correct

# For incorrectly classified images
python scripts/visualize_predictions.py \
    --model_path "path/to/your/vit_best.pth" \
    --model_name "facebook/dinov2-base" \
    --incorrect
Visualizing Attention Maps
To generate attention maps for Transformer-based models (ViT or DINOv2):

Bash

python scripts/visualize_attention.py \
    --model_path "path/to/your/vit_best.pth" \
    --model_name "facebook/dinov2-base" \
    --image_path "path/to/your/image.jpg" \
    --save_path "results/attention_map.png"


Model	Top-1 Validation Accuracy
DINOv2	79.64%
ViT-base	76.41%
ResNet50	59.33%

Export to Sheets
The DINOv2 model demonstrated the best performance and generalization, achieving 80.82% top-1 accuracy on the custom test set. For a full analysis, please see the project report.

Technologies Used
PyTorch

Hugging Face Transformers

torchvision

scikit-learn

Matplotlib & Seaborn

TensorBoard
