"""
s06/upload_dataset.py
----------------------
One-time script: uploads data/evals/golden_dataset.json to LangSmith as a
Dataset named LANGSMITH_DATASET_NAME (US-05). Re-run is safe -- it skips
creation if the dataset already exists, and skips examples already present.

Run once before eval.py:
    python upload_dataset.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "wealthdesk"))

from config import GOLDEN_DATASET_PATH, LANGSMITH_DATASET_NAME  # noqa: E402
from langsmith import Client  # noqa: E402


def main() -> None:
    with open(GOLDEN_DATASET_PATH, encoding="utf-8") as f:
        rows = json.load(f)

    client = Client()

    if client.has_dataset(dataset_name=LANGSMITH_DATASET_NAME):
        print(f"Dataset '{LANGSMITH_DATASET_NAME}' already exists. Skipping creation.")
        dataset = client.read_dataset(dataset_name=LANGSMITH_DATASET_NAME)
    else:
        dataset = client.create_dataset(
            dataset_name=LANGSMITH_DATASET_NAME,
            description="WealthDesk US-05 baseline golden dataset (40 Q&A pairs).",
        )
        print(f"Created dataset '{LANGSMITH_DATASET_NAME}'.")

    client.create_examples(
        inputs=[{"input": r["input"]} for r in rows],
        outputs=[{"expected_output": r["expected_output"]} for r in rows],
        metadata=[{"category": r["category"], "adversarial": r["adversarial"]} for r in rows],
        dataset_id=dataset.id,
    )
    print(f"Uploaded {len(rows)} examples to '{LANGSMITH_DATASET_NAME}'.")


if __name__ == "__main__":
    main()
