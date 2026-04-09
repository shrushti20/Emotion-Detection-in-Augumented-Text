import pandas as pd

FEWSHOT_PATH = "/home/hpc/v121ca/v121ca21/thesis_data/contarga_llm/fewshot_examples_static.csv"

def load_fewshot(path: str = FEWSHOT_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    # stable ordering for reproducibility
    return df.sort_values(["label", "row_id"]).reset_index(drop=True)

def fewshot_to_prompt_block(df: pd.DataFrame) -> str:
    blocks = []
    for _, r in df.iterrows():
        row_id = int(r["row_id"])
        label = str(r["label"])
        text = " ".join(str(r["text"]).split())
        blocks.append(
            f"[Example | source: CONTARGA subset row {row_id} | label: {label}]\n"
            f"Text: {text}\n"
        )
    return "\n".join(blocks)

def build_zero_shot_prompt(text: str, labels: list[str]) -> str:
    label_str = ", ".join(labels)
    return (
        "You are an emotion classification system.\n"
        "Choose exactly one label from the following list:\n"
        f"{{{label_str}}}\n\n"
        "Text:\n"
        f"{text}\n\n"
        "Answer:\n"
    )

def build_few_shot_prompt(text: str, labels: list[str], fewshot_df: pd.DataFrame | None = None) -> str:
    if fewshot_df is None:
        fewshot_df = load_fewshot()
    label_str = ", ".join(labels)
    examples = fewshot_to_prompt_block(fewshot_df)

    return (
        "You are an emotion classifier for argumentative text.\n"
        f"Choose exactly ONE label from: {label_str}\n\n"
        "Examples:\n"
        f"{examples}\n"
        "Now classify:\n"
        f"Text: {text}\n"
        "Answer with exactly one label.\n"
    )
