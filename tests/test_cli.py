"""
Unit tests for cli module.
"""

import pytest
from unittest.mock import patch
from cli import run_cli, run_jackson_derivative


class TestCLI:
    """Test cases for CLI functions."""

    def test_run_cli_callable(self):
        """Test that run_cli is callable."""
        # This is a basic test; full CLI testing would require mocking input
        assert callable(run_cli)

    @patch('builtins.input')
    @patch('builtins.print')
    def test_run_jackson_derivative_user_chooses_q(self, mock_print, mock_input):
        """Test that run_jackson_derivative allows user to choose q parameter."""
        # Mock user inputs: function, q, variable
        mock_input.side_effect = ['f(x) = x**2', '2.0', 'x']
        
        # Call the function
        run_jackson_derivative()
        
        # Check that input was called for q
        assert mock_input.call_count >= 3  # function, q, variable
        
        # Check that print was called with the result
        mock_print.assert_called_with('\n📘 Jackson q-derivative (q=2.0): 3x')