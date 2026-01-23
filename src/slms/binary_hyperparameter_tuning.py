import argparse
import random

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
from utils.custom_callbacks import SaveMetricsCallback
from utils.utils import (cartesian_product, CustomDataset, compute_metrics, load_config)


if __name__ == '__main__':

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    transformers.set_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("label", type=str, help="Label column to use for binary classification")
    args = parser.parse_args()
    label = args.label.upper()  # e.g., UCPI, CPV

    config = load_config()

    train_data = pd.read_csv(config["intention"]["data"]["train"], encoding='utf-8')
    validation_data = pd.read_csv(config["intention"]["data"]["validation"], encoding='utf-8')

    id2label = {0: "No", 1: "Yes"}
    label2id = {"No": 0, "Yes": 1}

    for experiment in config["intention"]["models"]:

        model = AutoModelForSequenceClassification.from_pretrained(
            experiment["model"],
            num_labels=2,
            id2label=id2label,
            label2id=label2id
        )
        tokenizer = AutoTokenizer.from_pretrained(experiment["model"])

        train_encodings = tokenizer(
            train_data['content'].tolist(),
            truncation=True,
            padding=True,
            max_length=256
        )
        val_encodings = tokenizer(
            validation_data['content'].tolist(),
            truncation=True,
            padding=True,
            max_length=256
        )

        train_dataset = CustomDataset(train_encodings, train_data[label].tolist())
        val_dataset = CustomDataset(val_encodings, validation_data[label].tolist())

        hyper_parameters_dict = {
            "eval_strategy": ["steps"],
            "learning_rate": [1e-5, 2e-5, 3e-5, 4e-5, 5e-5],
            "per_device_train_batch_size": [8],
            "per_device_eval_batch_size": [8],
            "num_train_epochs": [5],
            "warmup_ratio": [0.06, 0.1],
            "weight_decay": [0.01, 0.03, 0.05, 0.1],
            "fp16": [True],
            "metric_for_best_model": ["f1"],
            "load_best_model_at_end": [True],
            "save_total_limit": [2],
            "greater_is_better": [True],
            "save_strategy": ["steps"],
            "eval_steps": [50]
        }

        set_of_hyper_parameters = cartesian_product(hyper_parameters_dict)

        for ind, hyperparameters in enumerate(set_of_hyper_parameters):
            args = TrainingArguments(
                output_dir=f"output/{label}/{experiment['model']}/" + str(ind),
                eval_strategy=hyperparameters["eval_strategy"],
                learning_rate=hyperparameters["learning_rate"],
                per_device_train_batch_size=hyperparameters["per_device_train_batch_size"],
                per_device_eval_batch_size=hyperparameters["per_device_eval_batch_size"],
                num_train_epochs=hyperparameters["num_train_epochs"],
                warmup_ratio=hyperparameters["warmup_ratio"],
                weight_decay=hyperparameters["weight_decay"],
                fp16=hyperparameters["fp16"],
                metric_for_best_model=hyperparameters["metric_for_best_model"],
                load_best_model_at_end=hyperparameters["load_best_model_at_end"],
                save_total_limit=hyperparameters["save_total_limit"],
                greater_is_better=hyperparameters["greater_is_better"],
                save_strategy=hyperparameters["save_strategy"],
                eval_steps=hyperparameters["eval_steps"],
            )

            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
                callbacks=[SaveMetricsCallback(
                    csv_file_name=f"metrics/{label}/{experiment['model']}/validation/val_results_" + str(ind) + ".csv",
                    hyperparameters=hyperparameters)
                ]
            )

            trainer.train()
