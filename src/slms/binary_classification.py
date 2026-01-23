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
from utils.utils import (CustomDataset, compute_metrics, load_config)

if __name__ == '__main__':

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    transformers.set_seed(42)

    all_paths = [
        "config/config_CPV.yaml",
        "config/config_PASV.yaml",
        "config/config_PSSA.yaml",
        "config/config_UCPI.yaml",
        "config/config_UIOA.yaml"
    ]

    for path in all_paths[:1]:

        config = load_config(file_path=path)

        label = list(config.keys())[0]

        train_data = pd.read_csv(config[label]["data"]["train"], encoding='utf-8')
        validation_data = pd.read_csv(config[label]["data"]["validation"], encoding='utf-8')

        id2label = {0: "No", 1: "Yes"}
        label2id = {"No": 0, "Yes": 1}

        for experiment in config[label]["models"]:
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

            args = TrainingArguments(
                output_dir=f"output/{label}/{experiment['model']}/",
                eval_strategy=experiment["hyperparameters"]["eval_strategy"],
                learning_rate=experiment["hyperparameters"]["learning_rate"],
                per_device_train_batch_size=experiment["hyperparameters"]["per_device_train_batch_size"],
                per_device_eval_batch_size=experiment["hyperparameters"]["per_device_eval_batch_size"],
                num_train_epochs=experiment["hyperparameters"]["num_train_epochs"],
                warmup_ratio=experiment["hyperparameters"]["warmup_ratio"],
                weight_decay=experiment["hyperparameters"]["weight_decay"],
                fp16=experiment["hyperparameters"]["fp16"],
                metric_for_best_model=experiment["hyperparameters"]["metric_for_best_model"],
                load_best_model_at_end=experiment["hyperparameters"]["load_best_model_at_end"],
                save_total_limit=experiment["hyperparameters"]["save_total_limit"],
                greater_is_better=experiment["hyperparameters"]["greater_is_better"],
                save_strategy=experiment["hyperparameters"]["save_strategy"],
                eval_steps=experiment["hyperparameters"]["eval_steps"],
            )

            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
                callbacks=[SaveMetricsCallback(
                    csv_file_name=f"metrics/{label}/{experiment['model']}/validation/val_results.csv",
                    hyperparameters=experiment["hyperparameters"])
                ]
            )

            trainer.train()

            model_saved_path = f"output/{label}/final/{experiment['model']}/"

            trainer.save_model(model_saved_path)
            tokenizer.save_pretrained(model_saved_path)
