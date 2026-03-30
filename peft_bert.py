import argparse
import time
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np

def compute_metrics(eval_pred):
    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = metric_acc.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = metric_f1.compute(predictions=predictions, references=labels)["f1"]
    return {"accuracy": acc, "f1": f1}

class Adapter(torch.nn.Module):
    def __init__(self, hidden_size=768, bottleneck_size=64):
        super().__init__()
        self.down_project = torch.nn.Linear(hidden_size, bottleneck_size)
        self.activation = torch.nn.ReLU()
        self.up_project = torch.nn.Linear(bottleneck_size, hidden_size)

    def forward(self, x):
        return x + self.up_project(self.activation(self.down_project(x)))

def inject_adapters(model, hidden_size=768, bottleneck_size=64):
    for param in model.parameters():
        param.requires_grad = False
    
    for layer in model.bert.encoder.layer:
        # Wrap the output.dense of the FFN
        original_dense = layer.output.dense
        layer.output.dense = torch.nn.Sequential(
            original_dense,
            Adapter(hidden_size, bottleneck_size)
        )
    
    for param in model.classifier.parameters():
        param.requires_grad = True


def count_parameters(model):
    trainable_params, all_params = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        
        all_params += num_params
        if param.requires_grad:
            trainable_params += num_params
    
    return trainable_params, all_params

def main():
    parser = argparse.ArgumentParser(description="PEFT for BERT")
    parser.add_argument("--mode", type=str, required=True, choices=[
        "baseline", "lora", "adapter", "freeze_selective", "train_attention"
    ], help="The fine-tuning mode to run.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Train batch size")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max steps for debugging")
    args = parser.parse_args()

    print(f"Loading SST-2 Dataset...")
    dataset = load_dataset("glue", "sst2")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=128)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]
    
    if args.max_steps > 0:
        # Just reduce the dataset size for faster debugging if needed
        train_dataset = train_dataset.select(range(args.max_steps * args.batch_size))
        eval_dataset = eval_dataset.select(range(min(args.max_steps * args.batch_size, len(eval_dataset))))

    print(f"Loading Model: bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    # Apply the selected mode
    if args.mode == "lora":
        from peft import get_peft_model, LoraConfig, TaskType
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["query", "value"]  # common targets for BERT
        )
        model = get_peft_model(model, peft_config)
    elif args.mode == "adapter":
        inject_adapters(model)
    elif args.mode == "freeze_selective":
        # Freeze embeddings and the first 6 encoder layers
        for param in model.bert.embeddings.parameters():
            param.requires_grad = False
        for i in range(6):
            for param in model.bert.encoder.layer[i].parameters():
                param.requires_grad = False
    elif args.mode == "train_attention":
        # Freeze all except attention self and classifier
        for name, param in model.named_parameters():
            if "classifier" in name or "attention.self" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    trainable_params, all_params = count_parameters(model)
    print(f"Mode: {args.mode} | Trainable parameters: {trainable_params} | All parameters: {all_params} | Trainable %: {100 * trainable_params / all_params:.2f}")

    training_args = TrainingArguments(
        output_dir=f"./results_{args.mode}",
        learning_rate=2e-5 if args.mode == "baseline" else 1e-4, # higher LR for PEFT typically
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        logging_steps=100,
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds.")

    print("Evaluating...")
    eval_results = trainer.evaluate()
    print(f"Evaluation Results:")
    print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"F1 Score: {eval_results['eval_f1']:.4f}")
    
    # Save the report
    with open(f"report_{args.mode}.txt", "w") as f:
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Accuracy: {eval_results['eval_accuracy']:.4f}\n")
        f.write(f"F1 Score: {eval_results['eval_f1']:.4f}\n")
        f.write(f"Trainable Params: {trainable_params} ({(trainable_params/all_params)*100:.2f}%)\n")
        f.write(f"Total Params: {all_params}\n")
        f.write(f"Training Time: {training_time:.2f}s\n")

if __name__ == "__main__":
    main()
