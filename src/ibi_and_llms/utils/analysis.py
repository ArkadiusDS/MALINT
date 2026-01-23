import json
import re


def safe_parse(json_str):
    """
    Parse a JSON string and attempt to handle malformed JSON.
    """
    if isinstance(json_str, str):
        try:
            # Attempt to parse JSON normally
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try to fix the malformed JSON
            try:
                # Remove newlines and extra spaces
                json_str = re.sub(r"\s+", " ", json_str)

                # Match key-value pairs with optional nested dictionaries
                matches = re.findall(r'"([^"]+)"\s*:\s*({.*?}|".*?"|\d+|true|false|null)', json_str)

                # Construct a dictionary from matches
                parsed_dict = {}
                for key, value in matches:
                    # Attempt to parse the value if it's a nested dictionary
                    if value.startswith("{") and value.endswith("}"):
                        try:
                            parsed_dict[key] = json.loads(value)
                        except json.JSONDecodeError:
                            parsed_dict[key] = value  # Leave as-is if still invalid
                    else:
                        # Remove surrounding quotes from string values
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        parsed_dict[key] = value
                return parsed_dict
            except Exception as e:
                # If everything fails, log and return None
                print(f"Failed to parse JSON: {e}")
                return None
    else:
        return json_str


def label_encoding(row):
    if isinstance(row, bool):
        row = str(row)

    row = row.lower()
    if ("fake" in row and "real" in row) or ("true" in row and "false" in row):
        return 0
    elif ("fake" in row) or ("false" in row):
        return 1
    else:
        return 0


def encode_labels(dataset, column_names):
    dataset = dataset.copy()
    for column in column_names:
        dataset[column] = dataset[column].apply(lambda x: label_encoding(x))
    return dataset


def ensure_dictionary(dictionary):
    try:
        for key in dictionary:
            dictionary[key] = safe_parse(dictionary[key])
        return dictionary
    except TypeError:
        return None


def persuasion_check(dictionary):
    try:
        for key in dictionary:
            if dictionary[key]["is_used"] == "Yes":
                return 1

        return 0
    except KeyError:
        return 0
    except TypeError:
        return "Unknown"
