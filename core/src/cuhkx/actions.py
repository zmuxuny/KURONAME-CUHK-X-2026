"""Canonical action-name mappings shared by HAU and HARn tasks."""

from __future__ import annotations

from cuhkx.metadata import normalize_text

# The 40 HAU action names correspond to organizer HARn classes 0--39.
# Classes 24 and 26 deliberately remain distinct: phone use versus games.
HAU_CLASS_TEXT = {
    0: "washing face",
    1: "brushing teeth",
    2: "combing hair",
    3: "undressing",
    4: "wiping hands",
    5: "getting dressed",
    6: "drinking",
    7: "eating",
    8: "grabbing utensils",
    9: "pouring",
    10: "stirring",
    11: "peeling fruit",
    12: "sweeping",
    13: "mopping",
    14: "washing dishes",
    15: "wiping surface",
    16: "folding clothes",
    17: "typing on a keyboard",
    18: "writing",
    19: "calling",
    20: "checking the time",
    21: "reading",
    22: "turning a page",
    23: "listening to the music with headphones",
    24: "using a phone",
    25: "watching tv",
    26: "playing games",
    27: "taking a selfie",
    28: "running",
    29: "squats",
    30: "jumping jacks",
    31: "stretching",
    32: "standing up",
    33: "lying down",
    34: "sitting down",
    35: "lunges",
    36: "walking",
    37: "taking medicine",
    38: "massaging oneself",
    39: "checking body temperature",
}
HAU_TEXT_CLASS = {normalize_text(text): class_id for class_id, text in HAU_CLASS_TEXT.items()}


def hau_action_class(value: object) -> int:
    text = normalize_text(value)
    if text not in HAU_TEXT_CLASS:
        raise ValueError(f"unknown HAU action text: {value!r}")
    return HAU_TEXT_CLASS[text]
