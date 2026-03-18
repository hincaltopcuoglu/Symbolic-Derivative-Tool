"""
Unit tests for parser module.
"""

import pytest
from parser import FunctionParser


class TestFunctionParser:
    """Test cases for FunctionParser class."""

    def test_parse_function_definition_valid(self):
        """Test parsing a valid function definition."""
        func_str = "f(x, y) = x**2 + y"
        name, args, body = FunctionParser.parse_function_definition(func_str)
        assert name == "f"
        assert args == ["x", "y"]
        assert body == "x**2 + y"

    def test_parse_function_definition_no_name(self):
        """Test parsing an expression without function name."""
        func_str = "x**2 + y"
        name, args, body = FunctionParser.parse_function_definition(func_str)
        assert name is None
        assert args == []
        assert body == "x**2 + y"

    def test_parse_function_definition_single_arg(self):
        """Test parsing with single argument."""
        func_str = "g(t) = sin(t)"
        name, args, body = FunctionParser.parse_function_definition(func_str)
        assert name == "g"
        assert args == ["t"]
        assert body == "sin(t)"

    def test_parse_function_definition_invalid_format(self):
        """Test parsing invalid format raises ValueError."""
        func_str = "f x = x**2"
        with pytest.raises(ValueError, match="Invalid function definition format"):
            FunctionParser.parse_function_definition(func_str)

    def test_extract_variables_simple(self):
        """Test extracting variables from simple expression."""
        expr = "x + y"
        vars_ = FunctionParser.extract_variables(expr)
        assert vars_ == ["x", "y"]

    def test_extract_variables_with_functions(self):
        """Test extracting variables while ignoring function names."""
        expr = "sin(x) + cos(y) + z"
        vars_ = FunctionParser.extract_variables(expr)
        assert vars_ == ["x", "y", "z"]

    def test_extract_variables_constants_ignored(self):
        """Test that constants like pi, e are ignored."""
        expr = "x + pi + e + y"
        vars_ = FunctionParser.extract_variables(expr)
        assert vars_ == ["x", "y"]

    def test_extract_variables_duplicates_removed(self):
        """Test that duplicate variables are removed."""
        expr = "x + x + y"
        vars_ = FunctionParser.extract_variables(expr)
        assert vars_ == ["x", "y"]

    def test_extract_variables_numbers_ignored(self):
        """Test that numbers are not considered variables."""
        expr = "x + 2 + y"
        vars_ = FunctionParser.extract_variables(expr)
        assert vars_ == ["x", "y"]