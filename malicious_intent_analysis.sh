#!/opt/homebrew/bin/bash

# Define models (comment out the ones you don't want to use)
models=(
    "gpt-4.1-mini"
    "gpt-4o-mini"
    "gemini-2.0-flash"
    "meta-llama/Llama-3.3-70B-Instruct"
    "google/gemma-3-27b-it"
)


prompts_file_path="prompts/intent-based-inoculation/intention_knowledge_infusion.yaml"
method_type="icot_one_detailed_multistep"

# Define datasets
declare -a datasets=(
    "data/MALINT/test.csv"
    "data/CoAID/test.csv"
    "data/ISOTFakeNews/test.csv"
    "data/ECTF/test.csv"
    "data/EUDisinfo/test.csv"
)

# Function to run the script
run_script() {
    local dataset_file=$1
    local model=$2

    # Generate output file path
    local parent_dir
    parent_dir=$(basename "$(dirname "$dataset_file")")

    local output_file

    output_file="ibi_results/$model/$parent_dir/ICoT_One_Detailed_MultiStep/intention_analysis.csv"


    echo "Processing: $dataset_file with prompt to generate intent analysis on model $model..."
    # Run the Python script
    uv run src/ibi_and_llms/icot.py \
        -dataset_file "$dataset_file" \
        -model "$model" \
        -output "$output_file" \
        -prompts_file_path "$prompts_file_path" \
        -method_type "$method_type"
}

# Loop through datasets
for model in "${models[@]}"; do
  for dataset_file in "${datasets[@]}"; do
    run_script "$dataset_file" "$model"
  done
done

echo "All tasks completed."
