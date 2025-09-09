from typing import List, Tuple

SUSPICIOUS_KEYWORDS = ["transfer", "urgent", "password", "login", "click", "bank"]

def analyze_messages(messages: List[str]) -> Tuple[float, List[str]]:
    text = " ".join(messages).lower()
    flagged = [word for word in SUSPICIOUS_KEYWORDS if word in text]
    score = round(len(flagged) / len(SUSPICIOUS_KEYWORDS), 2)
    return score, flagged
