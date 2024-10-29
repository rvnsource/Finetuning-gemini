# %%
import traceback
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from optparse import Option
from typing import Optional, List, Callable, Any, Dict

import vertexai
import os
from datasets import load_dataset
from google.api_core.exceptions import ResourceExhausted
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from tqdm import tqdm
import pandas as pd
from google.cloud import aiplatform
import multiprocess as mp

from sklearn.model_selection import train_test_split
from vertexai.generative_models import GenerationConfig, HarmCategory, HarmBlockThreshold, GenerativeModel
import backoff

# %%
#######################
# Initialize Vertex AI
#######################
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/ravi/.config/gcloud/genai-434714-5b6098f8999f.json"
PROJECT_ID = "genai-434714"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}
vertexai.init(project=PROJECT_ID, location=LOCATION)



# %%
#############################
# Load and split the dataset
#############################
bbc_datasets = load_dataset("SetFit/bbc-news")

train = pd.DataFrame(bbc_datasets["train"])
test = pd.DataFrame(bbc_datasets["test"])

# Remove the row containing the specified text
text_to_remove = "jackson film  absolute disaster  a pr expert has told the michael jackson child abuse trial that the tv"
test = test[~test['text'].str.contains(text_to_remove, na=False)]

print(f"Training Dataset head: \n {train.head()}")
print(f"Testing Dataset head: \n {test.head()}")

print(f"Training Dataset shape: {train.shape}")
print(f"Testing Dataset shape: {test.shape}")

val, test = train_test_split(
    test, test_size=0.75, shuffle=True, random_state=2, stratify=test["label_text"]
)
print(f"Validation Dataset shape: {val.shape}")
print(f"Testing Dataset shape: {test.shape}")

print(f"Validation Dataset Labels' count: {val.label_text.value_counts()}")
print(f"Testing Dataset Lables' count: {test.label_text.value_counts()}")




# %%

system_prompt_zero_shot = """TASK:
Classify the text into ONLY one of the following classes [business, entertainment, politics, sport, tech].

CLASSES:
- business
- entertainment
- politics
- sport
- tech

INSTRUCTIONS
- Respond with ONLY one class.
- You MUST use the exact word from the list above.
- DO NOT create or use any other classes.
- CAREFULLY analyze the text before choosing the best-fitting category from [business, entertainment, politics, sport, tech].

"""

system_prompt_few_shot = f"""TASK:
Classify the text into ONLY one of the following classes [business, entertainment, politics, sport, tech].

CLASSES:
- business
- entertainment
- politics
- sport
- tech

INSTRUCTIONS:
- Respond with ONLY one class.
- You MUST use the exact word from the list above.
- DO NOT create or use any other classes.
- CAREFULLY analyze the text before choosing the best-fitting category from [business, entertainment, politics, sport, tech].

EXAMPLES:
- EXAMPLE 1:
    <user>
    {train.loc[train["label_text"] == "business", "text"].iloc[10]}
    <model>
    {train.loc[train["label_text"] == "business", "label_text"].iloc[10]}

- EXAMPLE 2:
    <user>
    {train.loc[train["label_text"] == "entertainment", "text"].iloc[10]}
    <model>
    {train.loc[train["label_text"] == "entertainment", "label_text"].iloc[10]}

- EXAMPLE 3:
    <user>
    {train.loc[train["label_text"] == "politics", "text"].iloc[10]}
    <model>
    {train.loc[train["label_text"] == "politics", "label_text"].iloc[10]}

- EXAMPLE 4:
    <user>
    {train.loc[train["label_text"] == "sport", "text"].iloc[10]}
    <model>
    {train.loc[train["label_text"] == "sport", "label_text"].iloc[10]}

- EXAMPLE 5:
    <user>
    {train.loc[train["label_text"] == "tech", "text"].iloc[10]}
    <model>
    {train.loc[train["label_text"] == "tech", "label_text"].iloc[10]}

"""



# %%
######################
# Model Configuration
######################
generation_config = GenerationConfig(max_output_tokens=10, temperature=0)
safety_settings = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
}

gem_pro_1_model_zero = GenerativeModel(
    "gemini-1.0-pro-002",  # e.g. gemini-1.5-pro-001, gemini-1.5-flash-001
    system_instruction=[system_prompt_zero_shot],
    generation_config=generation_config,
    safety_settings=safety_settings,
)



# %%
####################################################
# Utility Function: Batch Prediction (Parallelism)
####################################################

def backoff_hdlr(details) -> None:
    print(f"Backing off {details['wait']} seconds after {details['tries']} tries.")

def log_error(msg: str, *args: Any):
    mp.get_logger().error(msg, *args)
    raise Exception(msg)

def handle_exception_threading(f: Callable) -> Callable:
    def applicator(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            log_error(traceback.format_exc())

    return applicator

@handle_exception_threading
@backoff.on_exception(
    backoff.expo, ResourceExhausted, max_tries=30, on_backoff=backoff_hdlr
)
def _predict_message(message: str, model: GenerativeModel) -> Optional[str]:
    response = model.generate_content([message], stream=False)

    # TODO: Take care of LLM safety aspects
    response_dict = response.to_dict()

    # Check if "prompt_feedback" exists and has the key "block_reason"
    try:
        prompt_feedback = response_dict.get("prompt_feedback", {})
        block_reason = prompt_feedback.get("block_reason")

        if block_reason == "PROHIBITED_CONTENT":
            # Handle the case for prohibited content
            print("Block Reason: ", response.prompt_feedback.block_reason_message)
            print("Message: ", message)
            return response.prompt_feedback.block_reason_message
        else:
            # Handle other cases or continue processing
            return response.text
    except Exception as e:
        # Log the exception or handle errors accordingly
        print(f"An error occurred: {e}")

def batch_predict(
        messages: List[str],
        model: GenerativeModel,
        max_workers: int = 4
) -> List[Optional[str]]:

    predictions = list()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        partial_func = partial(_predict_message, model=model)
        for prediction in tqdm(pool.map(partial_func, messages), total=len(messages)):
            predictions.append(prediction)

    return predictions


def predictions_postprocessing(text: str) -> str:
    return text.strip().lower()

def evaluate_predictions(
    df: pd.DataFrame,
    target_column: str = "label_text",
    prediction_column: str = "prediction_labels",
    postprocessing: bool = True,
) -> Dict[str, float]:

    if postprocessing:
        df[prediction_column] = df[prediction_column].apply(predictions_postprocessing)

    y_true = df[target_column]
    y_pred = df[prediction_column]

    metrics_report = classification_report(y_true, y_pred, output_dict=True)
    overall_macro_f1_score = f1_score(y_true, y_pred, average="macro")
    overall_micro_f1_score = f1_score(y_true, y_pred, average="micro")
    weighted_precision = precision_score(y_true, y_pred, average="weighted")
    weighted_recall = recall_score(y_true, y_pred, average="weighted")

    metrics = {
        "accuracy": metrics_report["accuracy"],
        "weighted precision": weighted_precision,
        "weighted recall": weighted_recall,
        "macro f1 score": overall_macro_f1_score,
        "micro f1 score": overall_micro_f1_score
    }

    categories = ["business", "entertainment", "politics", "sport", "tech"]
    for category in categories:
        if category in metrics_report:
            metrics[f"{category}_f1_score"] = metrics_report[category]["f1-score"]

    return metrics



#######################################################################


# %%
messages_to_predict = test["text"].to_list()

# %%

predictions_zero_shot = batch_predict(
    messages=messages_to_predict, model=gem_pro_1_model_zero, max_workers=4
)

print(predictions_zero_shot)



# %%
# Create an Evaluation dataframe to store the predictions for all the experiments.
df_evals = test.copy()
df_evals["gem1.0-zero-shot_predictions"] = predictions_zero_shot

# Compute Evaluation Metrics for zero-shot prompt
metrics_zero_shot = evaluate_predictions(
    df_evals.copy(),
    target_column = "label_text",
    prediction_column = "gem1.0-zero-shot_predictions",
    postprocessing = True
)
print(metrics_zero_shot)




