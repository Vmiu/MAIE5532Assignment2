# MAIE5532 Assignment 2: Deep Learning Model Optimization Pipeline

## 🚀 Quick Start

### Prerequisites

Before running this code, ensure you have the following installed on your Linux machine:

#### 1. System Requirements

- **Operating System**: Linux (Ubuntu 20.04+ recommended)
- **Python**: 3.9 or higher
- **Git**: For cloning the repository
- **CUDA** (Optional but recommended for GPU acceleration): 11.8 or 12.x
- **cuDNN** (Optional but recommended for GPU acceleration): 8.6 or higher

#### 2. Install Git and required packages

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install git curl python3 python3-pip build-essential unzip 
```

#### 3. Install uv (Python Package Manager)

```bash
# Install uv using curl (recommended method)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to your PATH (add to ~/.bashrc or ~/.zshrc)
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Verify installation
uv --version
```

#### 4. Install Nvidia Driver

Please refer to [https://www.nvidia.com/en-us/drivers/]

### Installation Steps

1. **Clone the repository**

```bash
git clone https://github.com/Vmiu/MAIE5532Assignment2.git
cd MAIE5532Assignment2
```

1. **Install packages in virtual environment using uv**

```bash
uv sync
```

1. **Verify installation**

```bash
source .venv/bin/activate
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python -c "import tensorflow as tf; print('GPU available:', len(tf.config.list_physical_devices('GPU')) > 0)"
```

## 📁 Project Structure

``` bash
MAIE5532Assignment2/
├── part1_baseline.py              # Baseline CNN model
├── part2_cloud_optimization.py    # Cloud optimization techniques
├── part3_edge_optimization.py     # Edge optimization (pruning, quantization)
├── part4_deployment_pipeline.py   # Deployment pipeline
├── streamlined_analysis.py        # Model analysis utilities
├── logger_utils.py               # Logging utilities
├── pyproject.toml                # Project dependencies
├── requirements.txt              # Project dependencies (for not uv user)
├── uv.lock                      # Locked dependency versions
└── README.md                    # This file
```

## 🏃‍♂️ Running the Code

### Part 1: Baseline Model

```bash
uv run part1_baseline.py
```

This will:

- Create a baseline CNN model for CIFAR-10
- Train the model and save it
- Generate accuracy plots
- Log results to `part1_baseline/part1_terminal.log`

### Part 2: Cloud Optimization

```bash
uv run part2_cloud_optimization.py
```

This will:

- Implement mixed precision training
- Apply knowledge distillation
- Generate teacher and student models
- Log results to `part2_cloud_optimization/part2_terminal.log`

### Part 3: Edge Optimization

```bash
uv run part3_edge_optimization.py
```

This will:

- Apply pruning with different sparsity levels (0.25, 0.5, 0.75)
- Implement quantization (dynamic range, float16, full integer)
- Generate TensorFlow Lite models
- Create analysis plots
- Log results to `part3_edge_optimization/part3_terminal.log`

### Part 4: Deployment Pipeline

```bash
uv run part4_deployment_pipeline.py
```
