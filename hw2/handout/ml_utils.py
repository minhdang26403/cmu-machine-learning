"""
Common utility functions for machine learning tasks.

This module contains shared functions used across different ML algorithms,
including decision trees and data inspection utilities.
"""

import math
from collections import Counter


def calculate_entropy(probabilities: list[float]) -> float:
    """
    Calculate the entropy of a probability distribution.

    Args:
        probabilities: List of probability values (should sum to 1.0)

    Returns:
        float: The entropy value in bits (log base 2)

    Note:
        Zero probabilities are ignored to avoid log(0) errors.
        Empty probability lists will return 0.
    """
    return -sum(
        probability * math.log2(probability)
        for probability in probabilities
        if probability > 0
    )


def find_majority_label(label_counts: Counter[str]) -> str:
    """
    Find the majority label from a counter of label frequencies.

    In case of ties, returns the lexicographically largest label.

    Args:
        label_counts: Counter object with label frequencies

    Returns:
        str: The majority label
    """
    majority_label = ""
    majority_label_count = 0
    for label, count in label_counts.items():
        if count > majority_label_count or (
            count == majority_label_count and label > majority_label
        ):
            majority_label_count = count
            majority_label = label
    return majority_label
