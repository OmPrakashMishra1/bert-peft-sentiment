# Parameter-Efficient Fine-Tuning (PEFT) on BERT

**GitHub Repository**: [bert-peft-sentiment](https://github.com/OmPrakashMishra1/bert-peft-sentiment)

This repository contains the code and experiments for implementing and comparing various Parameter-Efficient Fine-Tuning strategies on `bert-base-uncased` using the SST-2 dataset.

## Installation

Ensure you have Python installed, then install the necessary dependencies:

```bash
pip install -r requirements.txt
```

## Running the Experiments

The core script is `peft_bert.py`. It accepts a `--mode` argument to run different fine-tuning configurations.

Available modes:
1. `baseline` (Full Fine-Tuning)
2. `lora` (Low-Rank Adaptation with PEFT)
3. `adapter` (Custom PyTorch Bottleneck Adapters)
4. `freeze_selective` (Freezes Bottom 6 Layers & Embeddings)
5. `train_attention` (Freezes everything except Self-Attention layers and Classifier)

### Example Usage:
```bash
python peft_bert.py --mode lora
```

### Additional Arguments:
- `--epochs`: Number of exact training phases (default: 3).
- `--batch_size`: The batch size to use (default: 32).
- `--max_steps`: An optional argument to limit the number of steps if you are running on a CPU and just want to verify the implementation logic quickly. Example: `--max_steps 10`.

## Batch Execution

To run all 5 experiments consecutively and automatically generate report files, execute the PowerShell script (make sure you configure the step numbers inside the script to your liking depending on if you have a GPU):
```powershell
.\run_experiments.ps1
```

## Reports
After execution, each configuration will output a `.txt` report file showing Accuracy, F1 Score, Trainable Parameters, and Execution Time. You can find the aggregated analysis in `PEFT_Project_Report.md`.
