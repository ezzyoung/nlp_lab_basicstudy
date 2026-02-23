import torch
import re
import string
import math


ARTICLES_REGEX = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return ARTICLES_REGEX.sub(" ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def best_subspan_exact_match(
    list_pred: list[str],
    list_answers: list[list[str]],
) -> dict:
    hit = 0
    for pred, answers in zip(list_pred, list_answers):
        pred = normalize_answer(pred)
        if len(pred) == 0:
            continue
        for answer in answers:
            answer = normalize_answer(answer)
            if answer in pred:
                hit += 1
                break
    return {
        'acc': hit/len(list_pred),
    }