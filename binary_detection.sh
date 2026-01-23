#!/opt/homebrew/bin/bash

# Define models (comment out the ones you don't want to use)
models=(
    "gpt-4.1-mini"
    "gpt-4o-mini"
    "gemini-2.0-flash"
    "meta-llama/Llama-3.3-70B-Instruct"
    "google/gemma-3-27b-it"
)

prompts_file_path="prompts/classification/binary_intention_detection.yaml"
dataset="data/MALINT/test.csv"

# Define datasets
declare -a method_types=(
    "UCPI"
    "CPV"
    "UIOA"
    "PSSA"
    "PASV"
)

# Function to run the script
run_script() {
    local dataset_file=$1
    local model=$2

    local output_file

    output_file="benchmark_results/$model/MALINT/$method_type/intention_binary_detection.csv"


    echo "Processing: $dataset_file for intent $method_type with prompt to binary detection with model $model..."
    # Run the Python script
    uv run src/ibi_and_llms/binary_detection.py \
        -dataset_file "$dataset" \
        -model "$model" \
        -output "$output_file" \
        -prompts_file_path "$prompts_file_path" \
        -method_type "$method_type"
}

# Loop through datasets
for model in "${models[@]}"; do
  for method_type in "${method_types[@]}"; do
    run_script "$method_type" "$model"
  done
done

echo "All tasks completed."
