```markdown
# Fine-Tuning ResNet50 on iNaturalist-12K

This script implements transfer learning using a pretrained ResNet50 model on the iNaturalist-12K dataset (10 classes)
with PyTorch and Weights & Biases.
Three different fine-tuning strategies are available.

---

## 📂 File Structure

```
DA6401_Assignment_2_part_B.py  # Complete implementation (model, training, data loading)
README.md                      # This documentation file
```

---

## ⚙️ Prerequisites

- **Python** 3.7+  
- **PyTorch** 1.8+  
- **torchvision** 0.9+  
- **tqdm** 4.x  
- **Weights & Biases** (`wandb`) 0.12+  

Install required packages via:
```bash
pip install torch torchvision tqdm wandb
```

---

## 🌱 Dataset Layout

The script expects the iNaturalist-12K dataset in the following structure:
```
<ROOT_DIR>/
├── train/    # Training images (subdirectories per class)
└── val/      # Validation images (subdirectories per class)
```

Default directory is `/kaggle/input/dataset2/inaturalist_12K` (Kaggle path). Override with `--root_dir` argument.

---

## 🚀 Usage

1. **Authenticate with W&B**:
   ```bash
   import wandb
   wandb.login(key="YOUR_WANDB_API_KEY")
   ```
   Paste your API key when prompted (or modify the script to include your key).

2. **Run training**:
   ```bash
   python DA6401_Assignment_2_part_B.py \
     --wandb_project YOUR_PROJECT_NAME \
     --root_dir /path/to/inaturalist_12K \
     --batch_size 32 \
     --no_of_epochs 10 \
     --learning_rate 0.001 \
     --augmentation_flag yes \
     --strategy_flag 2
   ```

3. **View all options**:
   ```bash
   python DA6401_Assignment_2_part_B.py --help
   ```

4. **Outputs**  
   - Training/validation metrics logged to Weights & Biases
   - Progress printed to console

---

## 🎛️ Command-Line Arguments

| Argument | Description | Type | Default | Choices |
|----------|-------------|------|---------|---------|
| `--wandb_project`, `-wp` | W&B project name | str | `DA6401_Assignment_2` | Any valid project name |
| `--root_dir`, `-rd` | Dataset root directory | str | `/kaggle/input/dataset2/inaturalist_12K` | Valid path |
| `--batch_size`, `-bS` | Batch size | int | 32 | 32, 64, 128 |
| `--no_of_epochs`, `-nE` | Training epochs | int | 10 | ≥1 |
| `--learning_rate`, `-lR` | Learning rate | float | 0.001 | >0 |
| `--augmentation_flag`, `-ag` | Use data augmentation | str | `no` | `yes`, `no` |
| `--strategy_flag`, `-st` | Fine-tuning strategy | int | 2 | 0, 1, 2 |

---

## 🏗️ Fine-Tuning Strategies

The script provides three transfer learning approaches:

1. **Strategy 0 (`--strategy_flag 0`)**:
   - Freezes all layers except final fully connected layer
   - Replaces final layer with new 10-class classifier

2. **Strategy 1 (`--strategy_flag 1`)**:
   - Freezes first 10 layers
   - Replaces final layer with new 10-class classifier
   - Allows remaining layers to be fine-tuned

3. **Strategy 2 (`--strategy_flag 2`)** [Default]:
   - Freezes all original ResNet50 layers
   - Adds new sequential block with:
     - Dense layer (256 neurons)
     - ReLU activation
     - Dropout (0.4)
     - Final 10-class output layer

---

## 📖 Notes

- All training metrics (accuracy/loss) are logged to Weights & Biases
- The script automatically uses GPU if available (falls back to CPU)
- Data augmentation includes:
  - Random horizontal flips
  - Random rotation (±10 degrees)
  - Color jitter
  - Random resized crops
- Default image size is 256×256 (resized from original)
- Uses Adam optimizer with CrossEntropyLoss
```
