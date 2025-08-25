"""
Dataset aggregation script for AI vs Human text classification.

This script downloads and consolidates multiple datasets from Kaggle and 
Hugging Face into a single CSV file. The final dataset contains text and 
labels (`generated`: 1 = AI-generated, 0 = human-written).
"""

import os
import logging
import pandas as pd
import kagglehub


# ----------------------------
# Logging configuration
# ----------------------------
logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ----------------------------
# Helper functions
# ----------------------------
def concat_dataset(main_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
  """Concatenate a new dataset to the main DataFrame."""
  return pd.concat([main_df, df], ignore_index=True)


def load_kaggle_csv(dataset: str, filename: str, rename: dict | None = None) -> pd.DataFrame:
  """Download a Kaggle dataset and return as DataFrame."""
  path = kagglehub.dataset_download(dataset)
  full_path = os.path.join(path, filename)
  df = pd.read_csv(full_path)
  if rename:
    df = df.rename(columns=rename)
  logger.info(f"Loaded Kaggle dataset: {dataset} ({len(df)} rows)")
  return df


def load_hf_csv(path: str, rename: dict | None = None, drop_cols: list[str] | None = None) -> pd.DataFrame:
  """Load a Hugging Face CSV dataset."""
  df = pd.read_csv(path)
  if drop_cols:
    df = df.drop(columns=drop_cols)
  if rename:
    df = df.rename(columns=rename)
  logger.info(f"Loaded HF CSV: {path} ({len(df)} rows)")
  return df


def load_hf_parquet(path: str, generated: int | None = None, rename: dict | None = None) -> pd.DataFrame:
  """Load a Hugging Face Parquet dataset."""
  df = pd.read_parquet(path)
  if rename:
      df = df.rename(columns=rename)
  if generated is not None:
      df = pd.DataFrame({"text": df["text"], "generated": [generated] * len(df)})
  logger.info(f"Loaded HF Parquet: {path} ({len(df)} rows)")
  return df


# ----------------------------
# Main aggregation
# ----------------------------
def build_dataset() -> pd.DataFrame:
  main_df = pd.DataFrame(columns=["text", "generated"])

  # Kaggle datasets
  main_df = concat_dataset(main_df, load_kaggle_csv("shanegerami/ai-vs-human-text", "AI_Human.csv"))
  main_df = concat_dataset(main_df, load_kaggle_csv("denvermagtibay/ai-generated-essays-dataset", "AI Generated Essays Dataset.csv"))
  main_df = concat_dataset(main_df, load_kaggle_csv("aknjit/human-vs-ai-text-classification-dataset", "your_dataset_5000.csv", rename={"label": "generated"}))
  main_df = concat_dataset(main_df, load_kaggle_csv("navjotkaushal/human-vs-ai-generated-essays", "balanced_ai_human_prompts.csv"))

  # Hugging Face datasets
  main_df = concat_dataset(main_df, load_hf_csv("hf://datasets/NabeelShar/ai_and_human_text/train_ai.csv", drop_cols=["prompt_name"]))
  main_df = concat_dataset(main_df, load_hf_parquet("hf://datasets/polygraf-ai/dmitva_filtered_human_text_15k_extended/data/train-00000-of-00001.parquet", generated=1))
  main_df = concat_dataset(main_df, load_hf_parquet("hf://datasets/polygraf-ai/dmitva_filtered_human_text_15k_generated/data/train-00000-of-00001.parquet", generated=1))
  main_df = concat_dataset(main_df, load_hf_parquet("hf://datasets/polygraf-ai/dmitva_filtered_human_text_15k_extended_generated/data/train-00000-of-00001.parquet", generated=1))
  main_df = concat_dataset(main_df, load_hf_parquet("hf://datasets/polygraf-ai/human-text-comp/data/split_1-00000-of-00001.parquet", generated=0))

  # Human/AI pairs dataset
  df = load_hf_csv("hf://datasets/dmitva/human_ai_generated_text/model_training_dataset.csv")
  human_df = pd.DataFrame({"text": df["human_text"], "generated": [0] * len(df)})
  ai_df = pd.DataFrame({"text": df["ai_text"], "generated": [1] * len(df)})
  main_df = concat_dataset(main_df, human_df)
  main_df = concat_dataset(main_df, ai_df)

  # Another dataset with labels
  main_df = concat_dataset(main_df, load_hf_parquet("hf://datasets/ardavey/human-ai-generated-text/data/train-00000-of-00001.parquet", rename={"label": "generated"}))

  # Final cleanup
  main_df = main_df.drop_duplicates().reset_index(drop=True)
  logger.info(f"Final dataset size: {len(main_df)} rows")

  return main_df


# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
  df = build_dataset()
  save_path = os.path.join(os.path.dirname(__file__), "..", "data", "data.csv")
  os.makedirs(os.path.dirname(save_path), exist_ok=True)
  df.to_csv(save_path, index=False)
  logger.info(f"Saved dataset to {save_path}")