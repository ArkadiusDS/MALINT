#!/opt/homebrew/bin/bash

# Define models (comment out the ones you don't want to use)
models=(
    "gpt-4.1-mini"
    "gpt-4o-mini"
    "gemini-2.0-flash"
    "meta-llama/Llama-3.3-70B-Instruct"
    "google/gemma-3-27b-it"
)

prompts_file_path="prompts/intent-based-inoculation/ibi_final_step.yaml"
method_type="icot_one_detailed_multistep"


define_datasets() {
    local model=$1
    if [[ "$model" == "gpt-4o-mini" ]]; then
        declare -a datasets=(
            "ibi_results/gpt-4o-mini/MALINT/ICoT_One_Detailed_MultiStep/intention_analysis.csv"
            "ibi_results/gpt-4o-mini/CoAID/ICoT_One_Detailed_MultiStep/intention_analysis.csv"
            "ibi_results/gpt-4o-mini/ISOTFakeNews/ICoT_One_Detailed_MultiStep/intention_analysis.csv"
            "ibi_results/gpt-4o-mini/ECTF/ICoT_One_Detailed_MultiStep/intention_analysis.csv"
            "ibi_results/gpt-4o-mini/EUDisinfo/ICoT_One_Detailed_MultiStep/intention_analysis.csv"
        )
    elif [[ "$model" == "meta-llama/Llama-3.3-70B-Instruct" ]]; then
        declare -a datasets=(
            "ibi_results/meta-llama/Llama-3.3-70B-Instruct/CoAID/ICoT_One_Detailed_MultiStep/intention_analysis.csv"
            "ibi_results/meta-llama/Llama-3.3-70B-Instruct/ISOTFakeNews/ICoT_One_Detailed_MultiStep/intention_analysis.csv"
            "ibi_results/meta-llama/Llama-3.3-70B-Instruct/MALINT/ICoT_One_Detailed_MultiStep/intention_analysis.csv"
            "ibi_results/meta-llama/Llama-3.3-70B-Instruct/ECTF/ICoT_One_Detailed_MultiStep/intention_analysis.csv"
            "ibi_results/meta-llama/Llama-3.3-70B-Instruct/EUDisinfo/ICoT_One_Detailed_MultiStep/intention_analysis.csv"
        )
    elif [[ "$model" == "gpt-4.1-mini" ]]; then
        declare -a datasets=(
            "ibi_results/gpt-4.1-mini/CoAID/ICoT_One_Detailed_MultiStep/intention_analysis.csv"
            "ibi_results/gpt-4.1-mini/ISOTFakeNews/ICoT_One_Detailed_MultiStep/intention_analysis.csv"
            "ibi_results/gpt-4.1-mini/MALINT/ICoT_One_Detailed_MultiStep/intention_analysis.csv"
            "ibi_results/gpt-4.1-mini/ECTF/ICoT_One_Detailed_MultiStep/intention_analysis.csv"
            "ibi_results/gpt-4.1-mini/EUDisinfo/ICoT_One_Detailed_MultiStep/intention_analysis.csv"
        )
    elif [[ "$model" == "google/gemma-3-27b-it" ]]; then
        declare -a datasets=(
            "ibi_results/google/gemma-3-27b-it/CoAID/ICoT_One_Detailed_MultiStep/intention_analysis.csv"
            "ibi_results/google/gemma-3-27b-it/ISOTFakeNews/ICoT_One_Detailed_MultiStep/intention_analysis.csv"
            "ibi_results/google/gemma-3-27b-it/MALINT/ICoT_One_Detailed_MultiStep/intention_analysis.csv"
            "ibi_results/google/gemma-3-27b-it/ECTF/ICoT_One_Detailed_MultiStep/intention_analysis.csv"
            "ibi_results/google/gemma-3-27b-it/EUDisinfo/ICoT_One_Detailed_MultiStep/intention_analysis.csv"
        )
    elif [[ "$model" == "gemini-2.0-flash" ]]; then
        declare -a datasets=(
            "ibi_results/gemini-2.0-flash/EUDisinfo/ICoT_One_Detailed_MultiStep/intention_analysis.csv"
            "ibi_results/gemini-2.0-flash/CoAID/ICoT_One_Detailed_MultiStep/intention_analysis.csv"
            "ibi_results/gemini-2.0-flash/ISOTFakeNews/ICoT_One_Detailed_MultiStep/intention_analysis.csv"
            "ibi_results/gemini-2.0-flash/MALINT/ICoT_One_Detailed_MultiStep/intention_analysis.csv"
            "ibi_results/gemini-2.0-flash/ECTF/ICoT_One_Detailed_MultiStep/intention_analysis.csv"
        )
    fi
    for dataset in "${datasets[@]}"; do
        echo "$dataset"
    done
}

declare -a prompt_types=("VaN" "Z-CoT" "DeF_Spec")

# Function to run the script
run_script() {
    local dataset_file=$1
    local prompt_type=$2
    local model=$3

    # Generate output file path
    local parent_dir
    parent_dir=$(dirname "$dataset_file")

    local output_file
    output_file="$parent_dir/$prompt_type/final.csv"


    echo "Processing: $dataset_file with prompt type $prompt_type on model $model..."
    uv run src/ibi_and_llms/icot.py \
        -dataset_file "$dataset_file" \
        -model "$model" \
        -output "$output_file" \
        -prompts_file_path "$prompts_file_path" \
        -prompt_type "$prompt_type" \
        -method_type "$method_type"
}

# Loop through prompt types and datasets
for model in "${models[@]}"; do
    # Dynamically define datasets based on the model using mapfile
    datasets=()
    mapfile -t datasets < <(define_datasets "$model")

    for prompt_type in "${prompt_types[@]}"; do
        for dataset_file in "${datasets[@]}"; do
            run_script "$dataset_file" "$prompt_type" "$model"
        done
    done
done

echo "All tasks completed."