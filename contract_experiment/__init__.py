"""Question-free generative and extractive context compression."""

CONDITIONS = (
    "original",
    "generative_general",
    "generative_priority",
    "extractive_general",
    "extractive_priority",
)
LABELS = ("Entailment", "Contradiction", "NotMentioned")
CONDITION_NAMES = dict(zip(CONDITIONS, (
    "Original", "생성형·일반 (Naive)", "생성형·보존 우선 (Contextualized)",
    "추출형·일반", "추출형·보존 우선",
)))
