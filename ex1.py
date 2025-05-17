#!/usr/bin/env python3
import argparse

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
import evaluate
import wandb
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--max_predict_samples",
        type=int,
        default=-1,
    )
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    if args.do_predict and not args.model_path:
        parser.error("--model_path is required when --do_predict is set")
    return args


def preprocess(examples, tokenizer):
    return tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        truncation=True,
        max_length=tokenizer.model_max_length,
    )


def compute_metrics(p):
    metric = evaluate.load("accuracy")
    preds = p.predictions.argmax(-1)
    return {
        "accuracy": metric.compute(predictions=preds, references=p.label_ids)[
            "accuracy"
        ]
    }


def train(args):
    raw = load_dataset("glue", "mrpc")
    train_ds = raw["train"]
    eval_ds = raw["validation"]
    if args.max_train_samples != -1:
        train_ds = train_ds.select(range(args.max_train_samples))
    if args.max_eval_samples != -1:
        eval_ds = eval_ds.select(range(args.max_eval_samples))

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_ds = train_ds.map(lambda ex: preprocess(ex, tokenizer), batched=True)
    eval_ds = eval_ds.map(lambda ex: preprocess(ex, tokenizer), batched=True)

    collator = DataCollatorWithPadding(tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    exp_name = (
        f"epoch_num_{args.num_train_epochs}_lr_{args.lr}_batch_size_{args.batch_size}"
    )
    wandb.init(project="anlp_ex1", name=exp_name)
    training_args = TrainingArguments(
        output_dir="./outputs",
        save_strategy="no",
        save_total_limit=1,
        logging_strategy="steps",
        logging_steps=1,
        logging_dir="./logs",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.lr,
        metric_for_best_model="accuracy",
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(f"models/{exp_name}")
    wandb.finish()


def predict(args):
    raw = load_dataset("glue", "mrpc")
    test_ds = raw["test"]
    if args.max_predict_samples != -1:
        test_ds = test_ds.select(range(args.max_predict_samples))

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    test_ds = test_ds.map(lambda ex: preprocess(ex, tokenizer), batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.eval()

    collator = DataCollatorWithPadding(tokenizer)
    trainer = Trainer(
        model=model,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    preds_output = trainer.predict(test_ds)
    preds = torch.tensor(preds_output.predictions).argmax(dim=-1).tolist()

    s1 = test_ds["sentence1"]
    s2 = test_ds["sentence2"]

    with open("predictions.txt", "w") as f:
        for sent1, sent2, label in zip(s1, s2, preds):
            f.write(f"{sent1}###{sent2}###{label}\n")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.do_train:
        train(args)
    if args.do_predict:
        predict(args)


if __name__ == "__main__":
    main()
