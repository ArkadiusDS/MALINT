import json
import os
import pandas as pd
from transformers import pipeline
from utils.utils import compute_metrics_for_test_data

# Load the test data and preprocess
test_data = pd.read_csv("data/MALINT/test.csv", encoding='utf-8')

all_labels = [
        "CPV",
        "PASV",
        "PSSA",
        "UCPI",
        "UIOA"
    ]

for label in all_labels:
    sources = [
        f"output/{label}/final/distilbert/distilbert-base-uncased/",
        f"output/{label}/final/google-bert/bert-large-uncased/",
        f"output/{label}/final/google-bert/bert-base-uncased/",
        f"output/{label}/final/FacebookAI/roberta-large/",
        f"output/{label}/final/FacebookAI/roberta-base/",
        f"output/{label}/final/microsoft/deberta-v3-base/",
        f"output/{label}/final/microsoft/deberta-v3-large/"
    ]

    for model_saved_path in sources:
        # Load the pipeline with CUDA
        classifier = pipeline(
            task="text-classification",
            model=model_saved_path,
            tokenizer=model_saved_path,
            device=0,
            top_k=None,
            truncation=True,
            padding=True,
            max_length=256
        )

        # Run pipeline on all content (batched)
        results = classifier(test_data["content"].tolist(), batch_size=32)

        # Apply thresholding
        test_data["prob_of_intent"] = [
            1 if next(score["score"] for score in result if score["label"] == "Yes") > 0.3 else 0
            for result in results
        ]

        res = compute_metrics_for_test_data(test_data[label], test_data["prob_of_intent"])

        # Create output directory if it doesn't exist
        model_name = model_saved_path.strip("/").split("/")[-1]

        output_path = f"metrics/{label}/test/{model_name}/"

        os.makedirs(output_path, exist_ok=True)

        # Save results to JSON
        with open(output_path + "results.json", "w") as f:
            json.dump(res, f, indent=4)
