"""
Parser module for symbolic derivative tool.
Handles parsing of function definitions and expressions.
"""

import re
from typing import List, Tuple, Optional


class FunctionParser:
    """Parses function definitions and extracts components."""

    @staticmethod
    def parse_function_definition(func_str: str) -> Tuple[Optional[str], List[str], str]:
        """
        Parse a function definition string like 'f(x, y) = x**2 + y'.

        Args:
            func_str: The function definition string.

        Returns:
            Tuple of (function_name, arguments_list, body_expression).
        """
        if "=" not in func_str:
            # Not a function definition, treat as expression
            return None, [], func_str

        left, right = func_str.split("=", 1)
        left = left.strip()
        right = right.strip()

        if "(" not in left or ")" not in left:
            raise ValueError("Invalid function definition format")

        name = left[:left.index("(")].strip()
        args_str = left[left.index("(")+1 : left.index(")")]
        args = [arg.strip() for arg in args_str.split(",") if arg.strip()]

        return name, args, right

    @staticmethod
    def extract_variables(expr: str) -> List[str]:
        """Extract variable names from an expression."""
        # Find all word-like tokens that could be variables
        candidates = set(re.findall(r'\b[a-zA-Z]\w*\b', expr))
        # Filter out known functions/constants
        known_non_vars = {'sin', 'cos', 'log', 'exp', 'sqrt', 'pi', 'e'}
        return sorted(candidates - known_non_vars)