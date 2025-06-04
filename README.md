# Continual-FM
Foundation Models as Class-Incremental Learners for Dermatological Image Classification

# Run Experiment Script

## 🧪 Script: `run_experiment.py`

### 🔧 Parameters

| Argument      | Type   | Description                                 |
|---------------|--------|---------------------------------------------|
| `--file_path` | string | Path to the CSV file containing embeddings. |
| `--data_name` | string | Dataset name (e.g. `ham`, `d7p`, `dmf`).    |
| `--model_name`| string | Name of the model (e.g. `derm`, `panderm`, `clip`). |

### ▶️ Usage

From the terminal, run the script with:

```bash
python run_experiment.py \
    --file_path <path_to_csv> \
    --data_name <data name> \
    --model_name <model name>
```

### ▶️ Example Usage

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
