"""
Differentiator module for symbolic derivative tool.
Contains the DerivativeVisitor class for AST-based differentiation.
"""

import ast
import re
from typing import List, Dict


class DerivativeVisitor(ast.NodeVisitor):
    """AST visitor for symbolic differentiation."""

    def __init__(self, func_str: str, diff_var: str, chain_expand: bool = False):
        """
        Initialize the differentiator.

        Args:
            func_str: Function definition string like 'f(x) = sin(x)'.
            diff_var: Variable to differentiate with respect to.
            chain_expand: Whether to expand chain rule terms.
        """
        self.chain_expand = chain_expand
        self.diff_var = diff_var
        self.terms: List[str] = []

        # Parse function
        from parser import FunctionParser  # Import here to avoid circular
        self.func_name, self.args, self.body = FunctionParser.parse_function_definition(func_str)

        if self.func_name is None:
            # Expression only, infer args
            self.args = FunctionParser.extract_variables(self.body)

        self.tree = ast.parse(self.body, mode="eval")

    def visit_BinOp(self, node: ast.BinOp) -> None:
        """Handle binary operations: +, -, *, /, **, etc."""
        if isinstance(node.op, ast.Add):
            self.visit(node.left)
            self.visit(node.right)

        elif isinstance(node.op, ast.Sub):
            left_expr = ast.unparse(node.left)
            left_visitor = DerivativeVisitor(f"f({', '.join(self.args)}) = {left_expr}", self.diff_var)
            left_visitor.visit(left_visitor.tree)
            left_term = " + ".join(left_visitor.terms) if left_visitor.terms else "0"

            right_expr = ast.unparse(node.right)
            right_visitor = DerivativeVisitor(f"f({', '.join(self.args)}) = {right_expr}", self.diff_var)
            right_visitor.visit(right_visitor.tree)
            right_term = " + ".join(right_visitor.terms) if right_visitor.terms else "0"

            self.terms.append(f"({left_term}) - ({right_term})")

        elif isinstance(node.op, ast.Pow):
            base = node.left
            exponent = node.right

            base_expr = ast.unparse(base)
            # exponent_expr = ast.unparse(exponent)

            # Get derivative of base
            base_visitor = DerivativeVisitor(f"f({', '.join(self.args)}) = {base_expr}", self.diff_var)
            base_visitor.visit(base_visitor.tree)
            base_deriv = " + ".join(base_visitor.terms) if base_visitor.terms else "0"

            # If exponent is constant, use chain rule
            if isinstance(exponent, ast.Constant) and base_deriv != "0":
                n = exponent.value
                self.terms.append(f"{n}*({base_expr})**({n - 1})*({base_deriv})")
            elif base_deriv == "0":
                self.terms.append("0")
            else:
                # Symbolic exponent not supported
                print("⚠️  Exponentiation with symbolic exponent not fully supported.")
                self.terms.append("0")

        elif isinstance(node.op, ast.Div):
            u = ast.unparse(node.left)
            v = ast.unparse(node.right)

            du = DerivativeVisitor(f"f({', '.join(self.args)}) = {u}", self.diff_var)
            du.visit(du.tree)
            du_term = " + ".join(du.terms) if du.terms else "0"

            dv = DerivativeVisitor(f"f({', '.join(self.args)}) = {v}", self.diff_var)
            dv.visit(dv.tree)
            dv_term = " + ".join(dv.terms) if dv.terms else "0"

            numerator = f"{v}*({du_term}) - {u}*({dv_term})"
            denominator = f"{v}**2"
            self.terms.append(f"({numerator})/({denominator})")

        elif isinstance(node.op, ast.FloorDiv):
            print("⚠️  Floor division (//) is not differentiable. Returning 0.")
            self.terms.append("0")

        elif isinstance(node.op, ast.Mod):
            print("⚠️  Modulo (%) is not differentiable in symbolic math. Returning 0.")
            self.terms.append("0")

        elif isinstance(node.op, ast.Mult):
            left = node.left
            right = node.right

            u = ast.unparse(left)
            v = ast.unparse(right)

            du = DerivativeVisitor(f"f({', '.join(self.args)}) = {u}", self.diff_var)
            du.visit(du.tree)
            du_term = " + ".join(du.terms) if du.terms else "0"

            dv = DerivativeVisitor(f"f({', '.join(self.args)}) = {v}", self.diff_var)
            dv.visit(dv.tree)
            dv_term = " + ".join(dv.terms) if dv.terms else "0"

            if du_term != "0" and dv_term != "0":
                left_term = f"({du_term})*{v}" if du_term != "0" else ""
                right_term = f"{u}*({dv_term})" if dv_term != "0" else ""
                combined = " + ".join(filter(None, [left_term, right_term]))
                self.terms.append(combined if combined else "0")
            elif du_term != "0":
                self.terms.append(f"{du_term}*{v}")
            elif dv_term != "0":
                self.terms.append(f"{u}*{dv_term}")
            else:
                self.terms.append("0")

    def visit_Name(self, node: ast.Name) -> None:
        """Handle variable names."""
        if node.id == self.diff_var:
            print(f"🚀 Found variable match: {node.id} == {self.diff_var} → 1")
            self.terms.append("1")
        else:
            print(f"🧩 Variable {node.id} is not diff target → 0")
            self.terms.append("0")

    def visit_Constant(self, node: ast.Constant) -> None:
        """Handle constants."""
        self.terms.append("0")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        """Handle unary operations like -x."""
        if isinstance(node.op, ast.USub):  # Unary minus
            operand_expr = ast.unparse(node.operand)
            inner = DerivativeVisitor(f"f({', '.join(self.args)}) = {operand_expr}", self.diff_var)
            inner.visit(inner.tree)
            expr = " + ".join(inner.terms) if inner.terms else "0"
            self.terms.append(f"-({expr})")
        else:
            self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Handle function calls like sin(x)."""
        if len(node.args) != 1:
            self.terms.append("0")
            return

        g_node = node.args[0]
        func_name = node.func.id
        g_expr = ast.unparse(g_node)

        # Recursively get derivative of inner expression g(x)
        g_prime_visitor = DerivativeVisitor(f"f({', '.join(self.args)}) = {g_expr}", self.diff_var, chain_expand=self.chain_expand)
        g_prime_visitor.visit(g_prime_visitor.tree)
        g_prime_expr = " + ".join(g_prime_visitor.terms)

        chain_rules = {
            'sin': lambda g: f"cos({g})",
            'cos': lambda g: f"-sin({g})",
            'log': lambda g: f"1/({g})",
            'ln': lambda g: f"1/({g})",
            'exp': lambda g: f"exp({g})"
        }

        if func_name not in chain_rules:
            print(f"⚠️ Unknown function '{func_name}' — treating derivative as 0.")
            self.terms.append("0")
            return

        outer_prime = chain_rules[func_name](g_expr)

        if self.chain_expand:
            self.terms.append(f"({outer_prime}) * ({g_prime_expr})")
        else:
            if g_prime_expr == "1":
                self.terms.append(outer_prime)
            elif g_prime_expr == "0":
                self.terms.append("0")
            else:
                self.terms.append(f"{outer_prime} * ({g_prime_expr})")

    @staticmethod
    def pretty(term: str) -> str:
        """Simplify and prettify a derivative expression."""
        superscripts = {
            '0': '⁰', '1': '¹', '2': '²', '3': '³',
            '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷',
            '8': '⁸', '9': '⁹'
        }

        # 1. Clean *1, 1*, *0
        term = re.sub(r'\*1\b', '', term)
        term = re.sub(r'\b1\*', '', term)
        term = re.sub(r'\*\(1\)', '', term)  # remove *(1)
        term = re.sub(r'\*0\b', '0', term)

        # 2. Remove +0 and 0+
        term = re.sub(r'\+ *0\b', '', term)
        term = re.sub(r'\b0 *\+', '', term)

        # 3. Convert 2*x → 2x
        term = re.sub(r'\b(\d+)\*([a-zA-Z(])', r'\1\2', term)

        # 4. Remove dangling * at end or before closing parenthesis
        term = re.sub(r'\*$', '', term)
        term = re.sub(r'\*(\))', r'\1', term)
        term = re.sub(r'\(\s*([^\)]+)\*\)', r'(\1)', term)  # handles (2x*) too
        # Remove * before + or -
        term = re.sub(r'\*\s*([+\-])', r'\1', term)
        # 5. Special simplification: (y*(1) - x*(0))/(y**2) → 1/y
        term = re.sub(r"\((\w+)\*\(1\) - \w+\*\(0\)\)/\(\1\*\*2\)", r"1/\1", term)

        term = re.sub(r'\b[a-zA-Z]\*\*0\b', '1', term)
        term = re.sub(r'[a-zA-Z]0\b', '', term)

        # Additional simplifications
        # Remove - - → +
        term = re.sub(r'-\s*-', '+', term)
        # Remove + + → +
        term = re.sub(r'\+\s*\+', '+', term)
        # Simplify -1* → -
        term = re.sub(r'-1\*([a-zA-Z(])', r'-\1', term)
        # Simplify 1* → 
        term = re.sub(r'\b1\*([a-zA-Z(])', r'\1', term)
        # Remove leading +
        term = re.sub(r'^\s*\+', '', term)
        # Simplify (term) if no operators inside
        term = re.sub(r'\(([a-zA-Z0-9]+)\)', r'\1', term)
        # Remove empty terms
        term = re.sub(r'\+\s*$', '', term)

        # Clean up .0 from integers
        term = re.sub(r'\.0', '', term)

        # 6. Convert **n to superscripts: x**2 → x²
        term = re.sub(r'([a-zA-Z])\s*\*\*\s*(\d+)',
                  lambda m: m.group(1) + ''.join(superscripts.get(ch, ch) for ch in m.group(2)),
                  term)

        # Simplify x¹ to x
        term = re.sub(r'([a-zA-Z])¹', r'\1', term)

        return term


def compute_gradient(func_str: str, variables: List[str]) -> List[str]:
    """
    Compute the gradient of a function with respect to given variables.
    
    Args:
        func_str: Function definition string like 'f(x,y) = x**2 + y'.
        variables: List of variables to differentiate with respect to.
    
    Returns:
        List of partial derivatives as pretty strings.
    """
    gradient = []
    
    for var in variables:
        visitor = DerivativeVisitor(func_str, var)
        visitor.visit(visitor.tree)
        
        filtered_terms = [t for t in visitor.terms if t != "0"]
        symbolic_expr = " + ".join(filtered_terms) if filtered_terms else "0"
        symbolic_pretty = DerivativeVisitor.pretty(symbolic_expr)
        
        gradient.append(symbolic_pretty)
    
    return gradient


class JacksonDerivativeVisitor(ast.NodeVisitor):
    """AST visitor for Jackson q-derivative computation."""

    def __init__(self, func_str: str, diff_var: str, q: float):
        """
        Initialize the Jackson q-derivative differentiator.

        Args:
            func_str: Function definition string like 'f(x) = sin(x)'.
            diff_var: Variable to differentiate with respect to.
            q: q parameter for Jackson derivative.
        """
        self.q = q
        self.diff_var = diff_var
        self.terms: List[str] = []

        # Parse function
        from parser import FunctionParser  # Import here to avoid circular
        self.func_name, self.args, self.body = FunctionParser.parse_function_definition(func_str)

        if self.func_name is None:
            # Expression only, infer args
            self.args = FunctionParser.extract_variables(self.body)

        self.tree = ast.parse(self.body, mode="eval")

    def visit_BinOp(self, node: ast.BinOp) -> None:
        """Handle binary operations for Jackson derivative."""
        # For Jackson derivative: D_q[f+g] = D_q f + D_q g
        if isinstance(node.op, ast.Add):
            self.visit(node.left)
            self.visit(node.right)
        elif isinstance(node.op, ast.Sub):
            self.visit(node.left)
            self.visit(node.right)
            # Negate the right terms
            self.terms[-len(self._get_terms_from_visit(node.right)):] = \
                ["-" + term for term in self.terms[-len(self._get_terms_from_visit(node.right)):]]
        elif isinstance(node.op, ast.Mult):
            # Product rule for Jackson derivative is more complex
            # For now, implement basic case
            left_expr = ast.unparse(node.left)
            right_expr = ast.unparse(node.right)
            
            # D_q[f*g] = f(qx)*D_q g + g(x)*D_q f + (q-1)x * D_q f * D_q g
            # This is complex, so for now we'll use a simplified approach
            self._add_jackson_term(left_expr, right_expr)
        elif isinstance(node.op, ast.Pow):
            base_expr = ast.unparse(node.left)
            if isinstance(node.right, ast.Constant) and isinstance(node.right.value, int):
                n = node.right.value
                if n > 0:
                    # D_q[x^n] = [ (qx)^n - x^n ] / [(q-1)x] = x^{n-1} * (q^n - 1)/(q-1)
                    q_n = self.q ** n
                    coeff = (q_n - 1) / (self.q - 1)
                    if n == 1:
                        self.terms.append(str(coeff))
                    else:
                        self.terms.append(f"{coeff}*x**{n-1}")
                else:
                    self.terms.append("0")
            else:
                # Symbolic exponent - not supported
                self.terms.append("0")
        else:
            # For other operations, return 0 for now
            self.terms.append("0")

    def visit_Name(self, node: ast.Name) -> None:
        """Handle variable names for Jackson derivative."""
        if node.id == self.diff_var:
            # D_q[x] = 1
            self.terms.append("1")
        else:
            # Constant with respect to differentiation variable
            self.terms.append("0")

    def visit_Constant(self, node: ast.Constant) -> None:
        """Handle constants for Jackson derivative."""
        self.terms.append("0")

    def visit_Call(self, node: ast.Call) -> None:
        """Handle function calls for Jackson derivative."""
        if len(node.args) != 1:
            self.terms.append("0")
            return

        func_name = node.func.id
        arg_expr = ast.unparse(node.args[0])

        # For Jackson derivative of functions, we need to know the q-analog
        # For now, implement basic trigonometric functions
        if func_name == 'sin':
            # D_q[sin(x)] = [sin(qx) - sin(x)] / [(q-1)x]
            # But we can leave it symbolic
            self.terms.append(f"(sin({self.q}*{arg_expr}) - sin({arg_expr})) / (({self.q} - 1)*{arg_expr})")
        elif func_name == 'cos':
            self.terms.append(f"(cos({self.q}*{arg_expr}) - cos({arg_expr})) / (({self.q} - 1)*{arg_expr})")
        elif func_name == 'exp':
            self.terms.append(f"(exp({self.q}*{arg_expr}) - exp({arg_expr})) / (({self.q} - 1)*{arg_expr})")
        else:
            # Unknown function
            self.terms.append(f"( {func_name}({self.q}*{arg_expr}) - {func_name}({arg_expr}) ) / (({self.q} - 1)*{arg_expr})")

    def _get_terms_from_visit(self, node: ast.AST) -> List[str]:
        """Helper to get terms from visiting a node."""
        initial_len = len(self.terms)
        self.visit(node)
        return self.terms[initial_len:]

    def _add_jackson_term(self, left_expr: str, right_expr: str) -> None:
        """Add Jackson derivative term for product."""
        # Simplified: assume one is constant
        if left_expr.replace('.', '').isdigit():
            # Constant times function
            const_val = float(left_expr) if '.' in left_expr else int(left_expr)
            self.terms.append(f"{const_val} * ( {right_expr.replace(self.diff_var, str(self.q) + '*' + self.diff_var)} - {right_expr} ) / (({self.q} - 1)*{self.diff_var})")
        elif right_expr.replace('.', '').isdigit():
            # Function times constant
            const_val = float(right_expr) if '.' in right_expr else int(right_expr)
            self.terms.append(f"{const_val} * ( {left_expr.replace(self.diff_var, str(self.q) + '*' + self.diff_var)} - {left_expr} ) / (({self.q} - 1)*{self.diff_var})")
        else:
            # Both functions - complex case
            self.terms.append(f"( {left_expr.replace(self.diff_var, str(self.q) + '*' + self.diff_var)} * {right_expr.replace(self.diff_var, str(self.q) + '*' + self.diff_var)} - {left_expr} * {right_expr} ) / (({self.q} - 1)*{self.diff_var})")