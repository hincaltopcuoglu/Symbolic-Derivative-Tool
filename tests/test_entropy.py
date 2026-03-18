"""
Unit tests for entropy module.
"""

import pytest
import math
from entropy import compute_shannon_entropy, compute_renyi_entropy, compute_tsallis_entropy


class TestShannonEntropy:
    """Test cases for Shannon entropy."""

    def test_uniform_distribution(self):
        """Test Shannon entropy for uniform distribution."""
        # 3 outcomes, each with p=1/3
        probabilities = [1/3, 1/3, 1/3]
        entropy = compute_shannon_entropy(probabilities)
        expected = math.log2(3)  # log2(3) ≈ 1.58496
        assert abs(entropy - expected) < 1e-6

    def test_deterministic(self):
        """Test Shannon entropy for deterministic distribution."""
        probabilities = [1.0, 0.0, 0.0]
        entropy = compute_shannon_entropy(probabilities)
        assert entropy == 0.0

    def test_two_outcomes(self):
        """Test Shannon entropy for two outcomes."""
        p = 0.3
        probabilities = [p, 1-p]
        entropy = compute_shannon_entropy(probabilities)
        expected = -(p * math.log2(p) + (1-p) * math.log2(1-p))
        assert abs(entropy - expected) < 1e-6

    def test_invalid_probabilities_negative(self):
        """Test error for negative probabilities."""
        with pytest.raises(ValueError, match="Probabilities must be between 0 and 1"):
            compute_shannon_entropy([-0.1, 1.1])

    def test_invalid_probabilities_sum(self):
        """Test error for probabilities that don't sum to 1."""
        with pytest.raises(ValueError, match="Probabilities must sum to 1"):
            compute_shannon_entropy([0.5, 0.3])

    def test_empty_list(self):
        """Test entropy for empty probability list."""
        entropy = compute_shannon_entropy([])
        assert entropy == 0.0


class TestRenyiEntropy:
    """Test cases for Renyi entropy."""

    def test_renyi_alpha_2(self):
        """Test Renyi entropy with α=2."""
        probabilities = [0.5, 0.5]
        entropy = compute_renyi_entropy(probabilities, 2)
        # H_2 = (1/(1-2)) log2(∑ p^2) = -log2(∑ p^2) = -log2(0.25 + 0.25) = -log2(0.5) = 1
        assert abs(entropy - 1.0) < 1e-6

    def test_renyi_alpha_0(self):
        """Test Renyi entropy with α=0 (special case)."""
        probabilities = [0.2, 0.3, 0.5]
        entropy = compute_renyi_entropy(probabilities, 0)
        # H_0 = log2(number of non-zero probabilities) = log2(3)
        expected = math.log2(3)
        assert abs(entropy - expected) < 1e-6

    def test_renyi_alpha_inf(self):
        """Test Renyi entropy with α=∞ (special case)."""
        probabilities = [0.1, 0.2, 0.7]
        entropy = compute_renyi_entropy(probabilities, float('inf'))
        # H_∞ = -log2(max(p)) = -log2(0.7)
        expected = -math.log2(0.7)
        assert abs(entropy - expected) < 1e-6

    def test_invalid_alpha_zero(self):
        """Test error for α < 0."""
        with pytest.raises(ValueError, match="Alpha must be greater than or equal to 0"):
            compute_renyi_entropy([0.5, 0.5], -1)

    def test_invalid_alpha_one(self):
        """Test error for α = 1."""
        with pytest.raises(ValueError, match="Alpha cannot be 1"):
            compute_renyi_entropy([0.5, 0.5], 1.0)


class TestTsallisEntropy:
    """Test cases for Tsallis entropy."""

    def test_tsallis_q_2(self):
        """Test Tsallis entropy with q=2."""
        probabilities = [0.5, 0.5]
        entropy = compute_tsallis_entropy(probabilities, 2)
        # S_2 = (1/(2-1)) ∑ [p^2 - p] = ∑ [p^2 - p] = [0.25 - 0.5] + [0.25 - 0.5] = -0.5
        assert abs(entropy - (-0.5)) < 1e-6

    def test_tsallis_q_0(self):
        """Test Tsallis entropy with q=0 (special case)."""
        probabilities = [0.2, 0.3, 0.5]
        entropy = compute_tsallis_entropy(probabilities, 0)
        # S_0 = n - 1 = 3 - 1 = 2
        assert entropy == 2.0

    def test_invalid_q_zero(self):
        """Test error for q < 0."""
        with pytest.raises(ValueError, match="q must be greater than or equal to 0"):
            compute_tsallis_entropy([0.5, 0.5], -1)

    def test_invalid_q_one(self):
        """Test error for q = 1."""
        with pytest.raises(ValueError, match="q cannot be 1"):
            compute_tsallis_entropy([0.5, 0.5], 1.0)