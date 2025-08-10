# Vision Transformers for Scene Recognition


## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/HitendraKawale/ViT_for_scene_recognition.git
    cd ViT_for_scene_recognition
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv vit_env
    source vit_env/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Dataset

The datasets are stored in the `data/` directory.

## Usage

### 1. Training

To train a model, run `train_from_config.py` with the desired configuration file. The script will automatically save the best model, a confusion matrix, and TensorBoard logs to the `results/runs/` directory.

* **Train DINOv2 (Best Model):**
    ```bash
    python scripts/train_from_config.py --config configs/dino_config.yaml
    ```

* **Train ViT-base:**
    ```bash
    python scripts/train_from_config.py --config configs/best_config.yaml
    ```

* **Train ResNet50:**
    ```bash
    python scripts/train_from_config.py --config configs/resnet_config.yaml
    ```

### 2. Evaluation on Custom Test Set

To evaluate a trained model on your custom test set, use the `evaluate.py` script.

```bash
python scripts/evaluate.py \
    --model_path "path/to/your/vit_best.pth" \
    --test_dir "data/custom_test_set/" \
    --model_name "name_of_your_model"
```

### 3. Visualising Predictions and Attention Maps

You can visualise predicted top correct classes or incorrect classes by the model using the visualize_prediction.py script

```bash
python scripts/visualize_predictions.py \
    --model_path "path/to/your/vit_best.pth" \
    --model_name "name_of_your_model" \
    --<correct/incorrect>
```

For Visualising Attention Maps for the models, you can use the visualize_attention.py script

```bash
python scripts/visualize_attention.py \
    --model_path "path/to/your/vit_best.pth" \
    --model_name "name_of_your_model" \
    --image_path "path/to/an/image.jpg" \
    --save_path "results/attention_map_name.png"
```

### 4. Comparing the Performance of all Models using TensorBoard

```bash
tesnsorboard --logdir ./results/run
```

