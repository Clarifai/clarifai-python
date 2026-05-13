def clean_text(value: str) -> str:
    return ' '.join(value.strip().split()).lower()
