"""
Entropy module for symbolic derivative tool.
Contains functions for computing various entropy measures.
"""

import math
from typing import List


def compute_shannon_entropy(probabilities: List[float]) -> float:
    """
    Compute Shannon entropy in bits.
    
    H(X) = -∑ p(x) log₂ p(x)
    
    Args:
        probabilities: List of probability values that sum to 1.
    
    Returns:
        Shannon entropy in bits.
    
    Raises:
        ValueError: If probabilities don't sum to 1 or contain invalid values.
    """
    if not probabilities:
        return 0.0
    
    # Validate probabilities
    if any(p < 0 or p > 1 for p in probabilities):
        raise ValueError("Probabilities must be between 0 and 1")
    
    prob_sum = sum(probabilities)
    if not math.isclose(prob_sum, 1.0, rel_tol=1e-6):
        raise ValueError(f"Probabilities must sum to 1, got {prob_sum}")
    
    # Compute entropy
    entropy = 0.0
    for p in probabilities:
        if p > 0:  # Avoid log(0)
            entropy -= p * math.log2(p)
    
    return entropy


def compute_renyi_entropy(probabilities: List[float], alpha: float) -> float:
    """
    Compute Renyi entropy of order alpha in bits.
    
    H_α(X) = (1/(1-α)) log₂ ∑ p(x)^α  for α ≠ 1
    As α → 1, reduces to Shannon entropy.
    
    Args:
        probabilities: List of probability values that sum to 1.
        alpha: Order parameter (must be > 0 and ≠ 1).
    
    Returns:
        Renyi entropy in bits.
    
    Raises:
        ValueError: If alpha <= 0 or alpha == 1, or invalid probabilities.
    """
    if math.isclose(alpha, 1.0, rel_tol=1e-6):
        raise ValueError("Alpha cannot be 1 (use Shannon entropy instead)")
    
    if not probabilities:
        return 0.0
    
    # Validate probabilities
    if any(p < 0 or p > 1 for p in probabilities):
        raise ValueError("Probabilities must be between 0 and 1")
    
    prob_sum = sum(probabilities)
    if not math.isclose(prob_sum, 1.0, rel_tol=1e-6):
        raise ValueError(f"Probabilities must sum to 1, got {prob_sum}")
    
    # Compute Renyi entropy
    if math.isclose(alpha, 0, rel_tol=1e-6):
        # Special case: H_0 = log₂(n) where n is number of non-zero probabilities
        non_zero_count = sum(1 for p in probabilities if p > 0)
        return math.log2(non_zero_count)
    elif alpha == float('inf'):
        # Special case: H_∞ = -log₂(max(p(x)))
        max_p = max(probabilities)
        return -math.log2(max_p)
    elif alpha < 0:
        raise ValueError("Alpha must be greater than or equal to 0")
    else:
        # General case
        sum_p_alpha = sum(p ** alpha for p in probabilities)
        entropy = (1.0 / (1.0 - alpha)) * math.log2(sum_p_alpha)
        return entropy


def compute_tsallis_entropy(probabilities: List[float], q: float) -> float:
    """
    Compute Tsallis entropy with parameter q.
    
    S_q(X) = (1/(q-1)) ∑ [p(x)^q - p(x)]  for q ≠ 1
    As q → 1, reduces to Shannon entropy.
    
    Args:
        probabilities: List of probability values that sum to 1.
        q: Tsallis parameter (must be > 0 and ≠ 1).
    
    Returns:
        Tsallis entropy.
    
    Raises:
        ValueError: If q <= 0 or q == 1, or invalid probabilities.
    """
    if math.isclose(q, 1.0, rel_tol=1e-6):
        raise ValueError("q cannot be 1 (use Shannon entropy instead)")
    
    if not probabilities:
        return 0.0
    
    # Validate probabilities
    if any(p < 0 or p > 1 for p in probabilities):
        raise ValueError("Probabilities must be between 0 and 1")
    
    prob_sum = sum(probabilities)
    if not math.isclose(prob_sum, 1.0, rel_tol=1e-6):
        raise ValueError(f"Probabilities must sum to 1, got {prob_sum}")
    
    # Compute Tsallis entropy
    if math.isclose(q, 0, rel_tol=1e-6):
        # Special case: S_0 = n - 1 where n is number of non-zero probabilities
        non_zero_count = sum(1 for p in probabilities if p > 0)
        return non_zero_count - 1
    elif q < 0:
        raise ValueError("q must be greater than or equal to 0")
    else:
        # General case
        sum_term = sum(p**q - p for p in probabilities)
        entropy = (1.0 / (q - 1.0)) * sum_term
        return entropy