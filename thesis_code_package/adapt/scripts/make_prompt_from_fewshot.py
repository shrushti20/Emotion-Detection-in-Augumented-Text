import pandas as pd

FEWSHOT_PATH = "/home/hpc/v121ca/v121ca21/thesis_data/contarga_llm/fewshot_examples_static.csv"

def load_fewshot(path: str = FEWSHOT_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    # stable ordering
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

def build_prompt(input_text: str, labels: list[str]) -> str:
    few = load_fewshot()
    examples = fewshot_to_prompt_block(few)
    label_str = ", ".join(labels)

    prompt = (
        "You are an emotion classifier for argumentative text.\n"
        f"Choose exactly ONE label from: {label_str}\n\n"
        "Examples:\n"
        f"{examples}\n"
        "Now classify:\n"
        f"Text: {input_text}\n"
        "Answer with exactly one label.\n"
    )
    return prompt

if __name__ == "__main__":
    few = load_fewshot()
    labels = sorted(few["label"].astype(str).unique().tolist())

    demo_text = "Mandatory retirement is unfair because it discriminates by age."
    prompt = build_prompt(demo_text, labels)

    print("=== LABELS ===")
    print(labels)
    print("\n=== PROMPT PREVIEW ===")
    print(prompt)
