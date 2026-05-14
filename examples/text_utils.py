def clean_text(value: str) -> str:
    return ' '.join(value.strip().split()).lower()


def concat(str1: str, str2: str) -> str:
    return str1 + str2
