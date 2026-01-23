# MALicious INTent Dataset and Inoculating LLMs for Enhanced Disinformation Detection

---

## Overview

Disinformation is not only about content but also the **underlying malicious intents**. This repository explores capabilities of SLMs such as BERT and LLMs like Llama 3.3 70B in binary detection of malicious intent categories and multilabel classification of all intent categories from proposed taxonomy in MALINT dataset.
Moreover, this codebase allows to explore the hypothesis that **equipping models with intent knowledge** can significantly boost their performance in disinformation detection. 

The codebase supports:

* **Binary & Multilabel Classification** of malicious intent.
  * Fine-tuning of **SLMs** (e.g., BERT) for binary and multilabel classification.
  * Prompting **LLMs** (e.g., Llama 3.3 70B) for binary and multilabel classification.
* Prompt-based **LLM experiments** for intent-base inoculation experiments:

  * Baseline disinformation detection.
  * Intent analysis (first step of IBI experiment).
  * Intent-informed disinformation detection (final step of IBI experiment).

A **new benchmark dataset**, **MALINT**, is introduced to support these tasks.

---

## Project Structure

<details>
<summary>Click to expand tree</summary>

```
├── data
│   ├── CoAID
│   │   ├── test.csv
│   │   ├── train.csv
│   │   └── validation.csv
│   ├── ECTF
│   │   ├── test.csv
│   │   ├── train.csv
│   │   └── validation.csv
│   ├── EUDisinfo
│   │   ├── original.csv
│   │   └── test.csv
│   ├── ISOTFakeNews
│   │   ├── test.csv
│   │   ├── train.csv
│   │   └── validation.csv
│   └── MALINT
│       ├── MALINT_benchmark.csv
│       ├── test.csv
│       ├── train.csv
│       └── valid.csv
├── prompts
│   ├── classification
│   │   ├── binary_intention_detection.yaml
│   │   └── multilabel_multiclass_classification.yaml
│   └── intent-based-inoculation
│       ├── baseline
│       │   └── simple_detection.yaml
│       ├── ibi_final_step.yaml
│       └── intention_knowledge_infusion.yaml
├── src
│   ├── ibi_and_llms
│   │   ├── binary_detection.py
│   │   ├── icot.py
│   │   └── utils
│   │       ├── analysis.py
│   │       └── utils.py
│   └── slms
│       ├── binary_classification.py
│       ├── binary_hyperparameter_tuning.py
│       ├── config
│       │   ├── config.yaml
│       │   ├── config_CPV.yaml
│       │   ├── config_PASV.yaml
│       │   ├── config_PSSA.yaml
│       │   ├── config_UCPI.yaml
│       │   └── config_UIOA.yaml
│       ├── data
│       │   └── MALINT
│       │       ├── test.csv
│       │       ├── train.csv
│       │       └── valid.csv
│       ├── intent_classification.py
│       ├── intent_hyperparameter_tuning.py
│       ├── predict_binary.py
│       ├── predict_multilabel.py
│       └── utils
│           ├── custom_callbacks.py
│           └── utils.py
├── binary_detection.sh
├── multilabel_multiclass_classification.sh
├── simple_detection.sh
├── malicious_intent_analysis.sh
├── run_icot_one_detailed_multistep.sh
├── README.md
├── LICENSE
├── pyproject.toml
└── uv.lock
```
</details>
---

## Shell Scripts Explained

Each shell script in the root directory was created for a specific part of the experimental pipeline:

| Script File                               | Purpose                                                                                                                                                                        |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `binary_detection.sh`                     | Runs **binary classification** of malicious intent using **five different LLMs**, one per intent category.                                                                     |
| `multilabel_multiclass_classification.sh` | Runs **multilabel classification** of intent (all five categories simultaneously) using LLMs.                                                                                  |
| `simple_detection.sh`                     | Launches **baseline disinformation detection** using prompting-only methods: **VaN**, **Z-CoT**, and **DeF-SpeC**. These do not use intent knowledge.                          |
| `malicious_intent_analysis.sh`            | Executes **Step 1** of the **Intent-Based Inoculation (IBI)** experiment. LLMs are used to **analyze and generate intent insights** from input text using intent-aware prompts. |
| `run_icot_one_detailed_multistep.sh`      | Executes **Step 2** of IBI: uses the intent analysis from Step 1 to perform **Intent-Augmented Reasoning** for improved disinformation detection via LLMs.               |

---


## Datasets

### 🆕 MALINT (MALicious INTent in Disinformation Dataset)

A novel dataset annotated with both **disinformation labels** and **malicious intent categories**, covering:

    1. Undermining the credibility of public institutions [UCPI]
    2. Changing political views [CPV]
    3. Undermining international organizations and alliances [UIOA]
    4. Promoting social stereotypes/antagonisms [PSSA]
    5. Promoting anti-scientific views [PASV]

Includes:

* `train.csv`
* `test.csv`
* `valid.csv`
* `MALINT_benchmark.csv`

> **Note:** To use this dataset, the articles must be scraped from the provided URLs. Alternatively, researchers may contact the main author of the paper introducing the dataset. Contact details are available at [amodzelewski.com](https://amodzelewski.com).

Other datasets (below datasets together with MALINT were used to evaluate Intent-base Inoculation):

* CoAID
* ECTF
* EUDisinfo
* ISOTFakeNews

---


## Experiments

### 1. Intent-Based Inoculation (IBI)

A two-step LLM experiment to test the effect of **intent augmented reasoning** for disinformation detection:

#### a. Step 1: Intent Analysis

```bash
bash malicious_intent_analysis.sh
```

* Uses LLMs to generate **malicious intent analysis** from news articles.

#### b. Step 2: Intent-Augmented Disinfo Detection

```bash
bash run_icot_one_detailed_multistep.sh
```

* Incorporates the generated intent into prompts for **disinformation detection**.

---

### 2. Prompt-Based Baselines (LLMs)

```bash
bash simple_detection.sh
```

Runs **baseline prompting methods**:

* **VaN**
* **Z-CoT**
* **DeF-SpeC**

---

### 3. Fine-tuned Models (SLMs)


Scripts for training and evaluating **Small Language Models (SLMs)** like BERT are located in:

```
src/slms/
```

### Tasks supported:

| Task                                                          | Script                                                               |
| ------------------------------------------------------------- | -------------------------------------------------------------------- |
| **Binary intent classification** (per category)               | `binary_classification.py`                                           |
| **Multilabel intent classification** (all categories at once) | `intent_classification.py`                                           |
| **Hyperparameter tuning**                                     | `binary_hyperparameter_tuning.py`, `intent_hyperparameter_tuning.py` |
| **Prediction (inference)**                                    | `predict_binary.py`, `predict_multilabel.py`                         |
| **Configuration files**                                       | Located in `src/slms/config/`                                        |
| **Callbacks and utilities**                                   | Located in `src/slms/utils/`                                         |

These scripts fine-tune SLMs with optimal hyperparameters using the provided MALINT dataset and intent labels. Each intent category has its own config for binary tasks, and a shared config exists for multilabel classification.


### Configuration Files

All configs with optimal hyperparameters and SLMs used in classification of intent are located in:

```
src/slms/config/
```

* `config.yaml`: used for **multilabel classification** with SLMs
* `config_*.yaml`: binary classification configs, one per intent category used for experiments with SLMs

---

This project uses `pyproject.toml` for dependency management (compatible with Poetry, pip, etc.).

---

## 📜 License

This codebase is licensed under the terms of the MIT License. Our novel MALINT dataset is licensed under CC BY 4.0 - Attribution 4.0 International.

---
