import argparse
import logging
import os
from datetime import datetime

# Import all necessary functions from the utils.
from utils.utils import (
    process_text,
    load_prompts,
    setup_logging,
    read_csv_file,
    sequential_processing_icot,
    parallel_processing_icot,
    parallel_text_processing,
    sequential_text_processing
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
        help="Prompting method to use. Options: 'simple_detection', "
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


def simple_detection_branch(args, df):
    """
    Executes the simple detection processing branch.
    Loads prompts and then processes the text using process_text.
    """
    logging.info(f"Executing simple detection branch with method_type: {args.method_type}")
    try:
        logging.info(f"Loading prompts with prompt_type: {args.prompt_type}")
        system_prompt, user_prompt = load_prompts(
            prompts_file_path=args.prompts_file_path,
            prompt_type=args.prompt_type
        )
    except:
        logging.error("Error loading prompts for simple detection.", exc_info=True)
        raise

    try:
        process_text(
            df=df.copy(),
            model=args.model,
            col_with_content="content",
            result_column="generated_analysis",
            output_file_path=args.output_file_path,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        logging.info(f"Simple detection processing completed successfully. Results saved to: {args.output_file_path}")
    except:
        logging.error("Error processing text in simple detection branch.", exc_info=True)
        raise


def icot_branch(args, df):
    """
    Executes the ICOT processing branch.
    Loads prompts (expecting three values) and then processes the text using either
    parallel or sequential processing based on the model.
    """
    logging.info(f"Executing ICOT branch with method_type: {args.method_type}")
    try:
        if args.prompt_type is not None:
            logging.info("Loading ICOT prompts with provided prompt_type.")
            system_prompt, user_prompt_1, user_prompt_2 = load_prompts(
                args.prompts_file_path, args.method_type, args.prompt_type
            )
        else:
            logging.info("Loading ICOT prompts without a prompt_type (intent analysis prompts).")
            system_prompt, user_prompt = load_prompts(
                args.prompts_file_path, args.method_type, None
            )
            user_prompt_1 = user_prompt
            user_prompt_2 = None
    except:
        logging.error("Error loading prompts for ICOT.", exc_info=True)
        raise

    try:
        if user_prompt_2:
            if args.model != "claude-3-haiku-20240307":
                parallel_processing_icot(
                    df=df.copy(),
                    col_with_content="content",
                    result_column="final_pred",
                    output_file=args.output_file_path,
                    model=args.model,
                    system_prompt=system_prompt,
                    user_part_1=user_prompt_1,
                    user_part_2=user_prompt_2,
                    analysis="generated_analysis",
                )
            else:
                sequential_processing_icot(
                    df=df.copy(),
                    col_with_content="content",
                    result_column="final_pred",
                    output_file=args.output_file_path,
                    model=args.model,
                    system_prompt=system_prompt,
                    user_part_1=user_prompt_1,
                    user_part_2=user_prompt_2,
                    analysis="generated_analysis",
                )
        else:

            if args.model != "claude-3-haiku-20240307":
                parallel_text_processing(
                    df=df.copy(),
                    col_with_content="content",
                    result_column="generated_analysis",
                    output_file=args.output_file_path,
                    model=args.model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt_1
                )
            else:
                sequential_text_processing(
                    df=df.copy(),
                    col_with_content="content",
                    result_column="generated_analysis",
                    output_file=args.output_file_path,
                    model=args.model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt_1,
                )
        logging.info(f"ICOT processing completed successfully. Results saved to: {args.output_file_path}")
    except:
        logging.error("Error during ICOT text processing.", exc_info=True)
        raise


def main():
    args = parse_arguments()

    # Configure logging
    configure_logging(args.model, args.method_type, args.dataset_file, args.output_file_path)

    # Read the dataset
    logging.info("Reading dataset...")
    try:
        df = read_csv_file(args.dataset_file)
    except:
        logging.error("Error reading the dataset file.", exc_info=True)
        raise

    # Decide which branch to execute based on the method_type.
    if args.method_type in ["simple_detection", "simple_detection_with_intention"]:
        simple_detection_branch(args, df)
    elif args.method_type in ["multilabel_multiclass"]:
        icot_branch(args, df)
    elif "icot" in args.method_type.lower():
        icot_branch(args, df)
    else:
        logging.error("Invalid method_type provided. Please use a valid method.")
        raise


if __name__ == "__main__":
    main()
