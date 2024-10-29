# %%
import traceback
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Optional, List, Callable, Any, Dict

import vertexai
import os
from datasets import load_dataset
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from tqdm import tqdm
import pandas as pd
from google.cloud import aiplatform

from sklearn.model_selection import train_test_split
from vertexai.generative_models import GenerationConfig, HarmCategory, HarmBlockThreshold, GenerativeModel

# %%
#######################
# Initialize Vertex AI
#######################
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/ravi/.config/gcloud/genai-434714-5b6098f8999f.json"
PROJECT_ID = "genai-434714"
LOCATION = "us-central1"
vertexai.init(project=PROJECT_ID, location=LOCATION)

# %%
#############################
# Load and split the dataset
#############################
bbc_datasets = load_dataset("SetFit/bbc-news")

train = pd.DataFrame(bbc_datasets["train"])
test = pd.DataFrame(bbc_datasets["test"])

# Remove specific row based on text
text_to_remove = "jackson film  absolute disaster  a pr expert has told the michael jackson child abuse trial that the tv"
test = test[~test['text'].str.contains(text_to_remove, na=False)]

print(f"Training Dataset shape: {train.shape}")
print(f"Testing Dataset shape: {test.shape}")

# Split test set into validation and test
val, test = train_test_split(
    test, test_size=0.75, shuffle=True, random_state=2, stratify=test["label_text"]
)
print(f"Validation Dataset shape: {val.shape}")
print(f"Testing Dataset shape: {test.shape}")

# %%
######################
# System Prompts
######################
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
- Use the exact word from the list above.
- Analyze the text carefully before choosing the best-fitting category.
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
    "gemini-1.0-pro-002",
    system_instruction=[system_prompt_zero_shot],
    generation_config=generation_config,
    safety_settings=safety_settings,
)

# %%
######################
# Prediction Functions
######################
def _predict_message(message: str, model: GenerativeModel) -> Optional[str]:
    try:
        response = model.generate_content([message], stream=False)
        response_dict = response.to_dict()

        # Handle prohibited content if necessary
        prompt_feedback = response_dict.get("prompt_feedback", {})
        block_reason = prompt_feedback.get("block_reason")

        if block_reason == "PROHIBITED_CONTENT":
            print(f"Blocked message: {message} - Reason: {response.prompt_feedback.block_reason_message}")
            return response.prompt_feedback.block_reason_message

        return response.text
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

def batch_predict_sequential(messages: List[str], model: GenerativeModel) -> List[Optional[str]]:
    predictions = []
    for message in tqdm(messages, total=len(messages), desc="Processing Messages"):
        prediction = _predict_message(message, model)
        predictions.append(prediction)
    return predictions

# %%
##############################
# Postprocessing and Evaluation
##############################
def predictions_postprocessing(text: Optional[str]) -> str:
    return text.strip().lower() if text else ""

def evaluate_predictions(
    df: pd.DataFrame, target_column: str, prediction_column: str, postprocessing: bool = True
) -> Dict[str, float]:
    if postprocessing:
        df[prediction_column] = df[prediction_column].apply(predictions_postprocessing)

    y_true = df[target_column]
    y_pred = df[prediction_column]

    metrics_report = classification_report(y_true, y_pred, output_dict=True)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    micro_f1 = f1_score(y_true, y_pred, average="micro")
    weighted_precision = precision_score(y_true, y_pred, average="weighted")
    weighted_recall = recall_score(y_true, y_pred, average="weighted")

    metrics = {
        "accuracy": metrics_report["accuracy"],
        "weighted precision": weighted_precision,
        "weighted recall": weighted_recall,
        "macro f1 score": macro_f1,
        "micro f1 score": micro_f1,
    }

    for category in ["business", "entertainment", "politics", "sport", "tech"]:
        if category in metrics_report:
            metrics[f"{category}_f1_score"] = metrics_report[category]["f1-score"]

    return metrics

# %%
######################
# Prediction and Evaluation
######################
messages_to_predict = test["text"].to_list()

# Generate predictions sequentially
predictions_zero_shot = batch_predict_sequential(messages_to_predict, gem_pro_1_model_zero)

# Store predictions in a DataFrame
df_evals = test.copy()
df_evals["gem1.0-zero-shot_predictions"] = predictions_zero_shot

# Evaluate predictions
metrics_zero_shot = evaluate_predictions(
    df_evals, target_column="label_text", prediction_column="gem1.0-zero-shot_predictions"
)

print(metrics_zero_shot)
