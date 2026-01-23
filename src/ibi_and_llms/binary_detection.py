import argparse
import logging
import os
from datetime import datetime

# Import all necessary functions from the utils.
from utils.utils import (
    load_prompts_binary_detection,
    setup_logging,
    read_csv_file,
    parallel_text_processing, sequential_text_processing
)


def configure_logging(model, method_type, dataset_file, output_file_path):
    """
    Configures logging: creates a log directory based on model and method_type and
    creates a log file with a datetime stamp.
    """
    log_dir = os.path.join(output_file_path.split("/")[0], "logging", model, method_type)
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(
        log_dir, f"logs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    )
    setup_logging(
        log_filename=log_filename,
        dataset_file=dataset_file,
        model=model,
        output_file_path=output_file_path,
    )


def parse_arguments():
    """
    Parses command-line arguments. The unified script handles both simple detection and ICOT modes.
    """
    parser = argparse.ArgumentParser(
        description="Unified processing script: Choose either a simple detection or an ICOT method."
    )
    parser.add_argument(
        "-dataset_file",
        type=str,
        required=True,
        help="Path to the test dataset file (CSV format).",
    )
    parser.add_argument(
        "-model", type=str, required=True, help="Model name to use for processing."
    )
    parser.add_argument(
        "-output_file_path",
        type=str,
        required=True,
        help="Output file path to save the results.",
    )
    parser.add_argument(
        "-prompts_file_path",
        type=str,
        required=True,
        help="Path to the prompts file (YAML format is also supported).",
    )
    parser.add_argument(
        "-method_type",
        type=str,
        required=True,
        help="Prompting method to use. Options: 'simple_detection', 'simple_detection_with_intention', "
             "'icot_one_detailed_multistep', etc.",
    )
    parser.add_argument(
        "-prompt_type",
        type=str,
        required=False,
        default=None,
        help="Optional prompt type parameter used by the prompt loader.",
    )
    return parser.parse_args()


def binary_detect_branch(args, df):
    """
    Executes the binary_detect_branch.
    Loads prompts (expecting three values) and then processes the text using either
    parallel or sequential processing based on the model.
    """
    logging.info(f"Executing binary_detect_branch with method_type: {args.method_type}")
    try:
        logging.info("Loading prompts for binary detection.")
        system_prompt, user_prompt = load_prompts_binary_detection(
            args.prompts_file_path, args.method_type
        )
    except:
        logging.error("Error loading prompts for binary detection.", exc_info=True)
        raise

    try:
        if args.model != "claude-3-haiku-20240307": # and "gemini" not in args.model:
            parallel_text_processing(
                df=df.copy(),
                col_with_content="content",
                result_column="pred",
                output_file=args.output_file_path,
                model=args.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
        else:
            sequential_text_processing(
                df=df.copy(),
                col_with_content="content",
                result_column="pred",
                output_file=args.output_file_path,
                model=args.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
    except:
        logging.error("Error during binary detection.", exc_info=True)
        raise


def main():
    args = parse_arguments()

    # Configure logging
    configure_logging(args.model, args.method_type, args.dataset_file, args.output_file_path)

    # Read the dataset
    logging.info("Reading dataset...")
    df = read_csv_file(args.dataset_file)
    logging.info("Binary detection...")
    binary_detect_branch(args, df)


if __name__ == "__main__":
    main()
