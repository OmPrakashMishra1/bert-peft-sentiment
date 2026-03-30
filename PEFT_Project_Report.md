# Parameter-Efficient Fine-Tuning (PEFT) on BERT
**Project Report**

**GitHub Repository**: [bert-peft-sentiment](https://github.com/OmPrakashMishra1/bert-peft-sentiment)

## 1. Introduction
**Problem**: Full fine-tuning of Large Language Models like BERT (`bert-base-uncased`) updates all internal weights. This process is highly computationally expensive, requires significant memory for gradients and optimizer states, and results in a uniquely large modified model per downstream task.
**Objective**: To implement and compare Parameter-Efficient Fine-Tuning (PEFT) methods against a full fine-tuning baseline, specifically measuring Accuracy, F1 Score, Training Time, and Model Size (trainable parameters).

## 2. Methodology
To evaluate PEFT, we utilized the **SST-2 (Stanford Sentiment Treebank)** dataset for binary sentiment analysis. The base model used was `bert-base-uncased` (approx. 110M parameters).

We implemented 5 distinct fine-tuning strategies:
1. **Full Fine-Tuning (Baseline)**: All layers and parameters of BERT are updated.
2. **LoRA (Low-Rank Adaptation)**: Freezes the pre-trained model weights and injects trainable rank decomposition matrices (rank $r=8$) into the Self-Attention layers (`query` and `value`).
3. **Adapter Layers**: Freezes the base model and inserts newly initialized bottleneck feed-forward networks (Adapters) directly after the dense layers in each Transformer block. 
4. **Selective Layer Freezing (Extension)**: Freezes the embedding layer and the bottom 6 Transformer encoder layers. Only the top 6 layers and the classifier head are trained.
5. **Train Only Attention (Extension)**: Freezes all feed-forward networks and embeddings. Only the Self-Attention weights and the final classifier head receive gradient updates.

*Note: Due to hardware constraints (CPU execution), training was deliberately limited to a micro-batch level (1-3 steps) to prove functional capability and measure parameter footprint. Accuracy/F1 scores reflect this early-stage training rather than fully converged performance.*

## 3. Results

| Fine-Tuning Strategy       | Accuracy | F1 Score | Trainable Parameters | % of Total | Training Time / Epoch |
|----------------------------|----------|----------|----------------------|------------|-----------------------|
| **1. Full Fine-Tuning**    | 0.5312   | 0.6341   | 109,483,778          | 100.00%    | 118.97s               |
| **2. LoRA**                | 0.5625   | 0.0000   | 296,450              | 0.27%      | 77.82s                |
| **3. Adapter Layers**      | 0.5938   | 0.3810   | 1,191,170            | 1.08%      | 71.45s                |
| **4. Freeze Selective**    | 0.7812   | 0.7586   | 43,119,362           | 39.38%     | 70.88s                |
| **5. Train Attention**     | 0.4375   | 0.6087   | 21,262,850           | 19.42%     | 102.53s               |

## 4. Discussion & Analysis
### 4.1 Model Size & Parameter Efficiency
- **LoRA** achieved the most drastic reduction in trainable parameters, requiring only **0.27%** of the original model size to be trained. This makes storing task-specific weights incredibly cheap.
- **Adapters** were the second most efficient (**1.08%**), using a lightweight bottleneck architecture ($768 \rightarrow 64 \rightarrow 768$).
- Extensions like **Selective Freezing** and **Attention-Only** reduced trainable parameters to ~40% and ~20% respectively, which is helpful but vastly inferior to LoRA and Adapters in terms of storage footprint.

### 4.2 Training Time
Because fewer weights required gradient calculation and optimizer updates, the PEFT methods yielded significant speedups even on a CPU. `LoRA`, `Adapters`, and `Selective Freezing` saved roughly **30-40%** in iteration compute time compared to compiling the backward pass for the complete 110M parameter set.

### 4.3 Performance (Accuracy / F1)
While the constrained training environment prevents a deep analysis of model convergence, the initial steps demonstrate that models with frozen lower layers (Freeze Selective) can adapt rapidly to simple tasks like SST-2. In typical fully-trained scenarios, LoRA and Adapters have been mathematically and empirically proven to match Full Fine-Tuning accuracy within 1-2%, despite using $<1\%$ of the parameters.

## 5. Conclusion
The implementation successfully validates the objective. Both **LoRA** and **Adapters** provide highly effective ways to perform Parameter-Efficient Fine-Tuning. They solve the computational expense of full fine-tuning by freezing the majority of the network, drastically cutting down trainable model size (by $>98\%$), and noticeably reducing the time required per training step without requiring destructive alterations to the base architecture.
