# Continual-FM
Foundation Models as Class-Incremental Learners for Dermatological Image Classification

# Abstract 

Class-Incremental Learning (CIL) aims to learn new classes over time without forgetting previously acquired knowledge. The emergence of foundation models (FM) pretrained on large datasets presents new opportunities for CIL by offering rich, transferable representations. However, their potential for enabling incremental learning in dermatology remains largely unexplored. In this paper, we systematically evaluate frozen FMs pretrained on large-scale skin lesion datasets for CIL in dermatological disease classification. We propose a simple yet effective approach where the backbone remains frozen, and a lightweight MLP is trained incrementally for each task. This setup achieves state-of-the-art performance without forgetting, outperforming regularization, replay, and architecture-based methods. To further explore the capabilities of frozen FMs, we examine zero-training scenarios using nearest mean classifiers with prototypes derived from their embeddings. Through extensive ablation studies, we demonstrate that this prototype-based variant can also achieve competitive results. Our findings highlight the strength of frozen FMs for continual learning in dermatology and support their broader adoption in real-world medical applications. 

---

# How To Run

### Setup Instructions

#### 1. Create a Virtual Environment

**For macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**For Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```
#### 2. Install Dependencies
After activating the virtual environment, run:
```bash
pip install -r requirements.txt
```
---

### Feature Extraction (Optional)

You have **two options** for obtaining the image embeddings needed for evaluation:  

#### ✅ Option 1: Use Precomputed Embeddings (Recommended)  


- Download all precomputed embeddings (`.csv` files) for all datasets and models directly from this link: [Download Embeddings](https://www.kaggle.com/datasets/mohammedelkhiat/foundation-models-embeddings/data)


- Once downloaded, place the files inside the `outputs/` directory and skip to the Run Experiment section.

---

#### Option 2: Extract Features Yourself

- Get a [Huggingface Token](https://huggingface.co/) to be able to use [Derm](https://huggingface.co/google/derm-foundation) and [CLIP](https://huggingface.co/openai/clip-vit-large-patch14) models.

- rename `.env.example` to `.env` and put your token as `HF_TOKEN=<your-token>` 

-  You can extract embeddings for any dataset and model combination using the `extract.py` script.

##### Script: `extract.py`

##### Parameters

| Argument      | Type   | Description                                 |
|---------------|--------|---------------------------------------------|
| `--data_name` | string | Dataset name (e.g. `ham`, `d7p`, `dmf`).    |
| `--model_name`| string | Name of the model (e.g. `derm`, `panderm`, `clip`). |

```bash
python extract.py \
    --data_name <data name> \
    --model_name <model name> 
```

This will automatically extract features and save the output as:

```bash
outputs/{model_name}_{data_name}.csv
```
---

### Run Experiment

#### Script: `run_experiment.py`

##### Parameters

| Argument      | Type   | Description                                 |
|---------------|--------|---------------------------------------------|
| `--file_path` (Optional)| string | Path to the CSV file containing embeddings. |
| `--data_name` | string | Dataset name (e.g. `ham`, `d7p`, `dmf`).    |
| `--model_name`| string | Name of the model (e.g. `derm`, `panderm`, `clip`). |

##### Usage

From the terminal, run the script with:

```bash
python run_experiment.py \
    --file_path <path_to_csv> \
    --data_name <data name> \
    --model_name <model name>
```

##### Example Usage

Experiment on `derm-foundation` model over `dmf` dataset:

```bash
python run_experiment.py \
    --file_path outputs/derm_dmf.csv \
    --data_name dmf \
    --model_name derm
```

#### ✅ Simple run (uses default file paths):
If you have already downloaded our embedding files and put them in the right place as required, you can run the script without specifying the file path:

```bash
python run_experiment.py \
    --data_name dmf \
    --model_name derm
```
