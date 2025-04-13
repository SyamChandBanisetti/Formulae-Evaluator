import re
from difflib import SequenceMatcher

correct_formulas = {
    "accuracy": "TP+TN/TP+TN+FP+FN",
    "precision": "TP/TP+FP",
    "recall": "TP/TP+FN",
    "f1_score": "2*precision*recall/precision+recall"
}

def clean_expr(expr):
    expr = expr.upper().replace(" ", "")
    expr = re.sub(r'[^A-Z0-9+*/()]', '', expr)
    return expr

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def evaluate_formulas(user_text):
    user_text = clean_expr(user_text)
    scores = {}
    total = 0

    for key, correct in correct_formulas.items():
        match_score = similar(user_text, correct)
        scores[key] = 1 if match_score > 0.75 else 0
        total += scores[key]

    scores["formatting"] = 1 if len(user_text) > 30 else 0  # example formatting check
    total += scores["formatting"]

    return scores, total
