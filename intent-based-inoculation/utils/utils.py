import os
import yaml
import time
import logging
import concurrent.futures
from typing import Tuple, Optional, Union, List, Dict
from dotenv import load_dotenv

import anthropic
import google.generativeai as genai
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# =============================================================================
# Environment Setup and Global Constants
# =============================================================================

# Load environment variables from .env file
load_dotenv()
TEMPERATURE = 0.0

# Retrieve API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")


# =============================================================================
# Helper Functions
# =============================================================================

def ensure_directory_exists(file_path: str) -> None:
    """Ensures that the parent directory for the given file path exists."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)


def update_dataframe_result(
        df: pd.DataFrame, index: int, system_prompt: str, user_prompt: str, completion: str, result_column: str
) -> None:
    """Updates the dataframe with the given result information."""
    df.at[index, "system_prompt"] = system_prompt
    df.at[index, "user_prompt"] = user_prompt
    df.at[index, result_column] = completion


def build_user_prompt(
        base_part: str, extra: Optional[str], suffix_part: str, text: str
) -> str:
    """Combines prompt parts and text to a complete prompt."""
    extra = extra or ""
    # Ensure proper spacing and punctuation as needed.
    return f"{base_part}{extra}\n{suffix_part} Text:{text}. Answer:"


# =============================================================================
# Setup and I/O Functions
# =============================================================================

def setup_logging(log_filename: str, dataset_file: str, model: str, output_file_path: str) -> None:
    """
    Sets up logging configuration and logs initial script parameters.
    """
    ensure_directory_exists(log_filename)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_filename)]
    )
    logging.info("Script started with the following parameters:")
    logging.info(f"Test file path: {dataset_file}")
    logging.info(f"Model: {model}")
    logging.info(f"Output file path: {output_file_path}")


def read_csv_file(dataset_file: str, encoding: str = "utf-8") -> pd.DataFrame:
    """
    Reads a CSV file into a Pandas DataFrame.
    """
    try:
        df = pd.read_csv(dataset_file, encoding=encoding)
        logging.info(f"Successfully read the file: {dataset_file}, Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error reading CSV file {dataset_file}: {e}", exc_info=True)
        raise


def load_prompts(
        prompts_file_path: str,
        method_type: Optional[str] = None,
        prompt_type: Optional[str] = None
) -> Union[Tuple[str, str], Tuple[str, str, str]]:
    """
    Loads prompts from a YAML file for a given method or prompt type.

    Returns:
        A tuple containing the system prompt and one or two user prompt parts.
    """
    try:
        with open(prompts_file_path, "r", encoding="utf-8") as file:
            prompts = yaml.safe_load(file)

        # Map from a simplified method_type to the key used in the YAML
        method_map = {
            "icot_one_detailed_multistep": "ICoT_One_Detailed_MultiStep",
            "multilabel_multiclass": "Multilabel_Multiclass",
            "simple_detection": "Simple_Detection"
        }

        if method_type:
            if method_type not in method_map:
                raise ValueError(f"Unsupported method type: {method_type}. Supported: {list(method_map.keys())}")
            method_key = method_map[method_type]
            if method_key not in prompts:
                raise KeyError(f"Missing prompts for method type: {method_type}")
            prompt_data = prompts[method_key]
            if prompt_type is not None and not pd.isna(prompt_type):
                prompt_data = prompt_data.get(prompt_type, prompt_data)
            system_prompt = prompt_data.get("system")
            user_prompt_1 = prompt_data.get("user") or prompt_data.get("user_part_1")
            user_prompt_2 = prompt_data.get("user_part_2")
            if system_prompt is None or user_prompt_1 is None:
                raise KeyError(f"Missing required prompt fields for method type: {method_type}")
            return (system_prompt, user_prompt_1, user_prompt_2) if user_prompt_2 else (system_prompt, user_prompt_1)

        if prompt_type and prompt_type in prompts:
            system_prompt = prompts[prompt_type].get("system")
            user_prompt = prompts[prompt_type].get("user")
            if system_prompt is None or user_prompt is None:
                raise KeyError(f"Missing required prompt fields for prompt type: {prompt_type}")
            return system_prompt, user_prompt

        raise ValueError("Either a valid method_type or prompt_type must be provided.")
    except (FileNotFoundError, yaml.YAMLError, KeyError, ValueError) as e:
        raise RuntimeError(f"Failed to load prompts: {e}")


def load_prompts_binary_detection(
        prompts_file_path: str,
        method_type: str
) -> Tuple[str, str]:
    """
    Loads prompts from a YAML file for a given method type.

    Returns:
        A tuple containing the system prompt and user prompt.
    """
    try:
        with open(prompts_file_path, "r", encoding="utf-8") as file:
            prompts = yaml.safe_load(file)

        system_prompt = prompts[method_type].get("system")
        user_prompt = prompts[method_type].get("user")
        if system_prompt is None or user_prompt is None:
            raise KeyError(f"Missing required prompt fields for prompt type: {method_type}")
        return system_prompt, user_prompt
    except (FileNotFoundError, yaml.YAMLError, KeyError, ValueError) as e:
        raise RuntimeError(f"Failed to load prompts: {e}")

# =============================================================================
# API Client Function
# =============================================================================


def client_instance(model: str) -> Union[OpenAI, anthropic.Anthropic]:
    """
    Returns an appropriate API client instance based on the model string.
    """
    if model in ["gpt-4o-mini", "gpt-4.1-mini"]:
        return OpenAI(api_key=OPENAI_API_KEY)
    elif model in ["meta-llama/Llama-3.3-70B-Instruct-Turbo", "meta-llama/Meta-Llama-3.1-8B-Instruct"]:
        return OpenAI(api_key=DEEPINFRA_API_KEY, base_url="https://api.deepinfra.com/v1/openai")
    elif model == "claude-3-haiku-20240307":
        return anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    else:
        raise ValueError(f"Unsupported model: {model}")


# =============================================================================
# Processing Functions
# =============================================================================

def process_text_sequentially(
        index: int,
        text: str,
        persuasion: str,
        model: str,
        df: pd.DataFrame,
        result_column: str,
        system_prompt: str,
        user_part_1: str,
        user_part_2: str
) -> Optional[Tuple[int, str]]:
    """
    Process a single row of the dataframe with an icot-style multi–step prompt.
    Combines parameters to form the user prompt and issues the API call.
    """
    try:
        user_prompt = build_user_prompt(user_part_1, persuasion, user_part_2, text)
        # For non-Gemini models, use the standard chat API call.
        client = client_instance(model=model)
        # Use different API call if needed (here using messages.create)
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=TEMPERATURE,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        # Assume response.content[0].text is the answer.
        answer = response.content[0].text
        update_dataframe_result(df, index, system_prompt, user_prompt, answer, result_column)
        return index, answer
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        time.sleep(3)  # Handle transient errors
        return None


def process_icot_multistep(
        index: int,
        text: str,
        persuasion: str,
        model: str,
        df: pd.DataFrame,
        result_column: str,
        system_prompt: str,
        user_part_1: str,
        user_part_2: str
) -> Optional[Tuple[int, str]]:
    """
    Processes a row using the ICoT multi–step workflow.
    Uses Gemini if applicable or a standard chat call otherwise.
    """
    try:
        user_prompt = build_user_prompt(user_part_1, persuasion, user_part_2, text)
        if model == "gemini-1.5-flash":
            genai.configure(api_key=GEMINI_API_KEY)
            gemini_model = genai.GenerativeModel(
                model_name="gemini-1.5-flash", system_instruction=system_prompt
            )
            response = gemini_model.generate_content(
                user_prompt,
                generation_config=genai.GenerationConfig(temperature=TEMPERATURE)
            )
            answer = response.text
        else:
            client = client_instance(model=model)
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=TEMPERATURE
            )
            answer = completion.choices[0].message.content
        update_dataframe_result(df, index, system_prompt, user_prompt, answer, result_column)
        return index, answer
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        time.sleep(3)
        return None


def process_text_with_model(
        index: int,
        text: str,
        model: str,
        system_prompt: str,
        user_prompt: str
) -> Dict[str, Union[int, str, None]]:
    """
    Processes a single text with a given model.
    Uses Gemini if applicable, otherwise makes a standard chat call.
    Returns a result dictionary.
    """
    try:
        full_prompt = f"{user_prompt} Text:{text}. Answer:"
        if model == "gemini-1.5-flash":
            genai.configure(api_key=GEMINI_API_KEY)
            gemini_model = genai.GenerativeModel(
                model_name="gemini-1.5-flash", system_instruction=system_prompt
            )
            response = gemini_model.generate_content(
                full_prompt,
                generation_config=genai.GenerationConfig(temperature=TEMPERATURE)
            )
            completion = response.text
        elif model == "claude-3-haiku-20240307":
            try:
                client = client_instance(model=model)
                completion = client.messages.create(
                    model=model,
                    max_tokens=4096,
                    temperature=TEMPERATURE,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": full_prompt}
                    ]
                )

                result = {
                    "index": index,
                    "system_prompt": system_prompt,
                    "user_prompt": full_prompt,
                    "completion": completion.content[0].text,
                }
                return result
            except Exception as e:
                print(f"Error processing row {index}: {e}")
                time.sleep(2)  # Delay before retrying in case of API issues
                return None

        else:
            client = client_instance(model=model)
            completion_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt},
                ],
                temperature=TEMPERATURE
            )
            completion = completion_response.choices[0].message.content
            return {
                "index": index,
                "system_prompt": system_prompt,
                "user_prompt": full_prompt,
                "completion": completion,
            }
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        time.sleep(2)  # Brief pause in case of repeated errors
        return {"index": index, "system_prompt": None, "user_prompt": None, "completion": None}


# =============================================================================
# Runner Functions (Sequential and Parallel)
# =============================================================================

def sequential_processing_icot(
        df: pd.DataFrame,
        col_with_content: str,
        result_column: str,
        output_file: str,
        model: str,
        system_prompt: str,
        user_part_1: str,
        user_part_2: str,
        analysis: str
) -> pd.DataFrame:
    """
    Sequentially process texts using the ICoT multi–step workflow.
    Respects a rate limit of 50 requests per minute.
    """
    # Create/clear result columns
    df["system_prompt"] = None
    df["user_prompt"] = None
    df[result_column] = None
    ensure_directory_exists(output_file)

    start_time = time.time()
    request_count = 0

    for idx, (text, analysis) in tqdm(
            enumerate(zip(df[col_with_content], df[analysis])),
            total=len(df)
    ):
        process_text_sequentially(idx, text, analysis, model, df, result_column, system_prompt, user_part_1,
                                  user_part_2)
        request_count += 1
        if request_count >= 50:
            elapsed = time.time() - start_time
            if elapsed < 60:
                wait_time = 60 - elapsed
                print(f"Rate limit reached. Waiting for {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            request_count = 0
            start_time = time.time()
    df.to_csv(output_file, index=False)
    return df


def parallel_processing_icot(
        df: pd.DataFrame,
        col_with_content: str,
        result_column: str,
        output_file: str,
        model: str,
        system_prompt: str,
        user_part_1: str,
        user_part_2: str,
        analysis: str
) -> pd.DataFrame:
    """
    Processes texts in parallel using ICoT multi–step function.
    """
    df["system_prompt"] = None
    df["user_prompt"] = None
    df[result_column] = None
    ensure_directory_exists(output_file)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for idx, (text, analysis) in enumerate(zip(df[col_with_content], df[analysis])):
            futures.append(executor.submit(
                process_icot_multistep, idx, text, analysis, model, df, result_column,
                system_prompt, user_part_1, user_part_2
            ))
        # Wait for all futures to complete.
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass

    df.to_csv(output_file, index=False)
    return df


def parallel_text_processing(
        df: pd.DataFrame,
        col_with_content: str,
        result_column: str,
        output_file: str,
        model: str,
        system_prompt: str,
        user_prompt: str
) -> None:
    """
    Parallel processing for the simpler single–prompt workflow.
    """
    df["system_prompt"] = None
    df["user_prompt"] = None
    df[result_column] = None
    ensure_directory_exists(output_file)

    results: List[Dict[str, Union[int, str, None]]] = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_text_with_model, idx, text, model, system_prompt, user_prompt)
            for idx, text in enumerate(df[col_with_content])
        ]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Thread failed with error: {e}")

    # Update DataFrame with results
    for result in results:
        update_dataframe_result(df, result["index"], result["system_prompt"], result["user_prompt"],
                                result["completion"], result_column)
    df.to_csv(output_file, index=False)


def sequential_text_processing(
        df: pd.DataFrame,
        col_with_content: str,
        result_column: str,
        output_file: str,
        model: str,
        system_prompt: str,
        user_prompt: str
) -> None:
    """
    Sequential processing for models (like Claude) that require rate limits.
    """
    df["system_prompt"] = None
    df["user_prompt"] = None
    df[result_column] = None
    ensure_directory_exists(output_file)

    results: List[Dict[str, Union[int, str, None]]] = []
    request_count = 0
    start_time = time.time()

    for idx, text in tqdm(enumerate(df[col_with_content]), total=len(df)):
        if request_count >= 50 and time.time() - start_time < 60:
            sleep_time = 60 - (time.time() - start_time)
            print(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
            request_count = 0
            start_time = time.time()
        result = process_text_with_model(idx, text, model, system_prompt, user_prompt)
        results.append(result)
        request_count += 1

    for res in results:
        update_dataframe_result(df, res["index"], res["system_prompt"], res["user_prompt"], res["completion"],
                                result_column)
    df.to_csv(output_file, index=False)


def process_text(
        df: pd.DataFrame,
        model: str,
        col_with_content: str,
        result_column: str,
        output_file_path: str,
        system_prompt: str,
        user_prompt: str
) -> None:
    """
    Processes text using the appropriate workflow based on the model.
    For models such as Claude a sequential version (with rate limiting) is used;
    for the others a parallel version is applied.
    """
    logging.info("Starting text processing.")
    try:
        if model == "claude-3-haiku-20240307":
            sequential_text_processing(
                df.copy(),
                col_with_content=col_with_content,
                result_column=result_column,
                output_file=output_file_path,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
        else:
            parallel_text_processing(
                df.copy(),
                col_with_content=col_with_content,
                result_column=result_column,
                output_file=output_file_path,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
        logging.info(f"Processing completed successfully. Results saved to: {output_file_path}")
    except Exception as e:
        logging.error("An error occurred during text processing.", exc_info=True)

# =============================================================================
# End of Module
# =============================================================================
