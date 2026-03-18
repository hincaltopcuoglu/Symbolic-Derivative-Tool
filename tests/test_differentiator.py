"""
Unit tests for differentiator module.
"""

import pytest
from differentiator import DerivativeVisitor, compute_gradient, JacksonDerivativeVisitor


class TestDerivativeVisitor:
    """Test cases for DerivativeVisitor class."""

    def test_constant_derivative(self):
        """Test derivative of a constant."""
        visitor = DerivativeVisitor("f(x) = 5", "x")
        visitor.visit(visitor.tree)
        assert visitor.terms == ["0"]

    def test_variable_derivative(self):
        """Test derivative of a variable."""
        visitor = DerivativeVisitor("f(x) = x", "x")
        visitor.visit(visitor.tree)
        assert visitor.terms == ["1"]

    def test_variable_not_target(self):
        """Test derivative of a variable not being differentiated."""
        visitor = DerivativeVisitor("f(x, y) = y", "x")
        visitor.visit(visitor.tree)
        assert visitor.terms == ["0"]

    def test_power_rule(self):
        """Test power rule: d/dx(x^n) = n*x^(n-1)."""
        visitor = DerivativeVisitor("f(x) = x**2", "x")
        visitor.visit(visitor.tree)
        assert visitor.terms == ["2*(x)**(1)*(1)"]

    def test_sum_rule(self):
        """Test sum rule: d/dx(f + g) = f' + g'."""
        visitor = DerivativeVisitor("f(x) = x + 3", "x")
        visitor.visit(visitor.tree)
        assert visitor.terms == ["1", "0"]

    def test_product_rule(self):
        """Test product rule: d/dx(f*g) = f'*g + f*g'."""
        visitor = DerivativeVisitor("f(x) = x * 2", "x")
        visitor.visit(visitor.tree)
        assert visitor.terms == ["1*2"]

    def test_quotient_rule(self):
        """Test quotient rule: d/dx(f/g) = (f'*g - f*g')/g^2."""
        visitor = DerivativeVisitor("f(x) = x / 2", "x")
        visitor.visit(visitor.tree)
        assert visitor.terms == ["(2*(1) - x*(0))/(2**2)"]

    def test_chain_rule_sin(self):
        """Test chain rule for sin(x)."""
        visitor = DerivativeVisitor("f(x) = sin(x)", "x")
        visitor.visit(visitor.tree)
        assert visitor.terms == ["cos(x)"]

    def test_chain_rule_exp(self):
        """Test chain rule for exp(x)."""
        visitor = DerivativeVisitor("f(x) = exp(x)", "x")
        visitor.visit(visitor.tree)
        assert visitor.terms == ["exp(x)"]

    def test_unary_minus(self):
        """Test derivative of -x."""
        visitor = DerivativeVisitor("f(x) = -x", "x")
        visitor.visit(visitor.tree)
        assert visitor.terms == ["-(1)"]

    def test_expression_only(self):
        """Test differentiating an expression without function name."""
        visitor = DerivativeVisitor("x**2 + 3*x", "x")
        visitor.visit(visitor.tree)
        # Should have terms for x**2 and 3*x
        assert len(visitor.terms) == 2
        assert "2*(x)**(1)*(1)" in visitor.terms
        assert "3*1" in visitor.terms

    def test_unknown_function(self):
        """Test unknown function returns 0."""
        visitor = DerivativeVisitor("f(x) = unknown(x)", "x")
        visitor.visit(visitor.tree)
        assert visitor.terms == ["0"]


class TestPretty:
    """Test cases for the pretty function."""

    def test_pretty_remove_star_one(self):
        """Test removing *1."""
        result = DerivativeVisitor.pretty("2*1")
        assert result == "2"

    def test_pretty_remove_one_star(self):
        """Test removing 1*."""
        result = DerivativeVisitor.pretty("1*x")
        assert result == "x"

    def test_pretty_coefficient_no_star(self):
        """Test converting 2*x to 2x."""
        result = DerivativeVisitor.pretty("2*x")
        assert result == "2x"

    def test_pretty_superscript(self):
        """Test converting **2 to ²."""
        result = DerivativeVisitor.pretty("x**2")
        assert result == "x²"

    def test_pretty_remove_plus_zero(self):
        """Test removing +0."""
        result = DerivativeVisitor.pretty("x + 0")
        assert result == "x "

    def test_pretty_remove_star_before_plus(self):
        """Test removing * before +."""
        result = DerivativeVisitor.pretty("2x* + 3")
        assert result == "2x+ 3"

    def test_pretty_simplify_minus_one(self):
        """Test simplifying -1*x to -x."""
        result = DerivativeVisitor.pretty("-1*x")
        assert result == "-x"

    def test_pretty_remove_leading_plus(self):
        """Test removing leading +."""
        result = DerivativeVisitor.pretty("+x + y")
        assert result == "x + y"

    def test_pretty_complex_expression(self):
        """Test pretty on a complex expression."""
        expr = "2*(x)**(2 - 1)*(1) + 0"
        result = DerivativeVisitor.pretty(expr)
        assert result == "2x**(2 - 1) "


class TestGradient:
    """Test cases for gradient computation."""

    def test_gradient_simple(self):
        """Test gradient of x**2 + y**2."""
        func_str = "x**2 + y**2"
        variables = ["x", "y"]
        grad = compute_gradient(func_str, variables)
        assert grad == ["2x", "2y"]

    def test_gradient_mixed(self):
        """Test gradient of x*y + x."""
        func_str = "x*y + x"
        variables = ["x", "y"]
        grad = compute_gradient(func_str, variables)
        assert grad == ["y + 1", "x"]

    def test_gradient_single_var(self):
        """Test gradient with single variable."""
        func_str = "x**3"
        variables = ["x"]
        grad = compute_gradient(func_str, variables)
        assert grad == ["3x²"]

    def test_gradient_trigonometric(self):
        """Test gradient with trigonometric functions."""
        func_str = "sin(x) + cos(y)"
        variables = ["x", "y"]
        grad = compute_gradient(func_str, variables)
        assert grad == ["cosx", "-siny"]


class TestJacksonDerivative:
    """Test cases for Jackson q-derivative."""

    def test_jackson_constant(self):
        """Test Jackson derivative of a constant."""
        visitor = JacksonDerivativeVisitor("f(x) = 5", "x", 2.0)
        visitor.visit(visitor.tree)
        assert visitor.terms == ["0"]

    def test_jackson_variable(self):
        """Test Jackson derivative of x."""
        visitor = JacksonDerivativeVisitor("f(x) = x", "x", 2.0)
        visitor.visit(visitor.tree)
        assert visitor.terms == ["1"]

    def test_jackson_power_x2(self):
        """Test Jackson derivative of x^2."""
        visitor = JacksonDerivativeVisitor("f(x) = x**2", "x", 2.0)
        visitor.visit(visitor.tree)
        # D_2[x^2] = (4 - 1)/(2-1) * x = 3x
        expected = ["3.0*x**1"]
        assert visitor.terms == expected

    def test_jackson_power_x3(self):
        """Test Jackson derivative of x^3."""
        visitor = JacksonDerivativeVisitor("f(x) = x**3", "x", 2.0)
        visitor.visit(visitor.tree)
        # D_2[x^3] = (8 - 1)/(2-1) * x^2 = 7x^2
        expected = ["7.0*x**2"]
        assert visitor.terms == expected

    def test_jackson_sin(self):
        """Test Jackson derivative of sin(x)."""
        visitor = JacksonDerivativeVisitor("f(x) = sin(x)", "x", 2.0)
        visitor.visit(visitor.tree)
        expected = ["(sin(2.0*x) - sin(x)) / ((2.0 - 1)*x)"]
        assert visitor.terms == expected

    def test_jackson_sum(self):
        """Test Jackson derivative of sum."""
        visitor = JacksonDerivativeVisitor("f(x) = x + 1", "x", 2.0)
        visitor.visit(visitor.tree)
        # D_q[x + 1] = D_q x + D_q 1 = 1 + 0
        assert "1" in visitor.terms
        assert "0" in visitor.terms