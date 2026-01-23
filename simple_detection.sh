#!/opt/homebrew/bin/bash

# Define models (comment out the ones you don't want to use)
models=(
    "gpt-4.1-mini"
    "gpt-4o-mini"
    "gemini-2.0-flash"
    "meta-llama/Llama-3.3-70B-Instruct"
    "google/gemma-3-27b-it"
)

prompts_file_path="prompts/intent-based-inoculation/baseline/simple_detection.yaml"
method_type="simple_detection"

# Define common datasets
declare -a datasets=(
    "data/CoAID/test.csv"
    "data/ISOTFakeNews/test.csv"
    "data/MALINT/test.csv"
    "data/EUDisinfo/test.csv"
    "data/ECTF/test.csv"
)

declare -a prompt_types=("VaN" "Z-CoT" "DeF_Spec")
# Function to run the script
run_script() {
    local dataset_file=$1
    local prompt_type=$2
    local model=$3

    # Generate output file path
    local parent_dir
    parent_dir=$(basename "$(dirname "$dataset_file")")
    local output_file

    output_file="ibi_results/$model/$parent_dir/Simple_Detection/$prompt_type/simple_detection.csv"


    echo "Processing: $dataset_file with prompt type $prompt_type on model $model..."
    uv run src/ibi_and_llms/icot.py \
        -dataset_file "$dataset_file" \
        -model "$model" \
        -output "$output_file" \
        -prompts_file_path "$prompts_file_path" \
        -prompt_type "$prompt_type" \
        -method_type "$method_type"
}  # <-- Ensure this closing brace is here.

# Main loop to execute tasks
for model in "${models[@]}"; do
    for prompt_type in "${prompt_types[@]}"; do
        for dataset_file in "${datasets[@]}"; do
            run_script "$dataset_file" "$prompt_type" "$model"
        done
    done
done

echo "All tasks completed."
