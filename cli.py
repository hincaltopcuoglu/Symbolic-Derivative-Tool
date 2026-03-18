"""
CLI module for symbolic derivative tool.
Handles user interaction for symbolic differentiation.
"""

from differentiator import DerivativeVisitor, compute_gradient, JacksonDerivativeVisitor
from entropy import compute_shannon_entropy, compute_renyi_entropy, compute_tsallis_entropy
from parser import FunctionParser


def run_cli() -> None:
    """Run the symbolic differentiation tool."""
    print("📘 Symbolic Derivative Tool")
    print("Choose an operation:")
    print("1. Symbolic Differentiation")
    print("2. Lagrange Multiplier Method")
    print("3. Advanced Mathematical Computations")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        run_differentiation_menu()
    elif choice == "2":
        run_lagrange()
    elif choice == "3":
        run_advanced_math()
    else:
        print("❌ Invalid choice")


def run_differentiation_menu() -> None:
    """Menu for different types of differentiation."""
    print("\n📘 Symbolic Differentiation")
    print("Choose differentiation type:")
    print("1. Standard derivative")
    print("2. Jackson q-derivative")
    
    diff_choice = input("Enter choice (1 or 2): ").strip()
    
    if diff_choice == "1":
        run_standard_derivative()
    elif diff_choice == "2":
        run_jackson_derivative()
    else:
        print("❌ Invalid choice")


def run_standard_derivative() -> None:
    """Run basic symbolic differentiation."""
    print("\n📘 Standard Derivative")
    print("Enter a function (e.g., f(x) = sin(x)) and the variable to differentiate with respect to.")
    print("Example: Function: f(x) = x**2 + 3*x")
    print("         Variable: x")

    func_str = input("Function: ").strip()
    diff_var = input("Variable to differentiate with respect to: ").strip()

    try:
        visitor = DerivativeVisitor(func_str, diff_var)
        visitor.visit(visitor.tree)

        filtered_terms = [t for t in visitor.terms if t != "0"]
        symbolic_expr = " + ".join(filtered_terms) if filtered_terms else "0"
        symbolic_pretty = DerivativeVisitor.pretty(symbolic_expr)

        print(f"\n📘 Symbolic derivative: {symbolic_pretty}")

    except Exception as e:
        print(f"❌ Error: {e}")


def run_jackson_derivative() -> None:
    """Run Jackson q-derivative computation."""
    print("\n📘 Jackson q-Derivative")
    print("Jackson derivative: D_q f(x) = [f(qx) - f(x)] / [(q-1)x]")
    print("Enter a function and the q parameter.")
    print("Example: Function: f(x) = x**2")
    print("         q: 2")

    func_str = input("Function: ").strip()
    q_value = input("q parameter: ").strip()
    diff_var = input("Variable to differentiate with respect to: ").strip()

    try:
        q = float(q_value)
        visitor = JacksonDerivativeVisitor(func_str, diff_var, q)
        visitor.visit(visitor.tree)

        filtered_terms = [t for t in visitor.terms if t != "0"]
        symbolic_expr = " + ".join(filtered_terms) if filtered_terms else "0"
        symbolic_pretty = DerivativeVisitor.pretty(symbolic_expr)

        print(f"\n📘 Jackson q-derivative (q={q}): {symbolic_pretty}")

    except ValueError:
        print("❌ Invalid q parameter. Must be a number.")
    except Exception as e:
        print(f"❌ Error: {e}")


def run_lagrange() -> None:
    """Run Lagrange multiplier method."""
    print("\n📘 Lagrange Multiplier Method")
    print("Find critical points of f(x,y,...) subject to g(x,y,...)=c")
    print("Example: Objective: f(x,y) = x**2 + y**2")
    print("         Constraint: g(x,y) = x + y - 1")
    
    obj_func = input("Objective function f: ").strip()
    constraint_func = input("Constraint function g: ").strip()
    
    try:
        # Parse functions to get variables
        _, obj_args, _ = FunctionParser.parse_function_definition(obj_func)
        _, const_args, _ = FunctionParser.parse_function_definition(constraint_func)
        
        if obj_args is None:
            obj_args = FunctionParser.extract_variables(obj_func)
        if const_args is None:
            const_args = FunctionParser.extract_variables(constraint_func)
            
        # Get all variables
        all_vars = sorted(set(obj_args + const_args))
        
        if not all_vars:
            print("❌ No variables found")
            return
            
        print(f"\n📘 Variables: {', '.join(all_vars)}")
        
        # Compute gradients
        grad_f = compute_gradient(obj_func, all_vars)
        grad_g = compute_gradient(constraint_func, all_vars)
        
        print("\n📘 Lagrange Equations:")
        print("∇f = λ ∇g")
        for i, var in enumerate(all_vars):
            f_deriv = DerivativeVisitor.pretty(grad_f[i])
            g_deriv = DerivativeVisitor.pretty(grad_g[i])
            print(f"∂f/∂{var} = λ ∂g/∂{var}")
            print(f"{f_deriv} = λ {g_deriv}")
            print()
            
        print("📘 Constraint:")
        print(f"g(x,y,...) = c")
        print(f"{constraint_func} = c")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def run_advanced_math() -> None:
    """Menu for advanced mathematical computations."""
    print("\n📘 Advanced Mathematical Computations")
    print("Choose computation:")
    print("1. Shannon Entropy")
    print("2. Renyi Entropy")
    print("3. Tsallis Entropy")
    
    math_choice = input("Enter choice (1-3): ").strip()
    
    if math_choice == "1":
        run_shannon_entropy()
    elif math_choice == "2":
        run_renyi_entropy()
    elif math_choice == "3":
        run_tsallis_entropy()
    else:
        print("❌ Invalid choice")


def run_shannon_entropy() -> None:
    """Compute Shannon entropy."""
    print("\n📘 Shannon Entropy")
    print("H(X) = -∑ p(x) log₂ p(x)")
    print("Enter probability distribution as comma-separated values.")
    print("Example: 0.5, 0.3, 0.2")
    
    prob_str = input("Probabilities: ").strip()
    
    try:
        probabilities = [float(p.strip()) for p in prob_str.split(",")]
        entropy = compute_shannon_entropy(probabilities)
        print(f"\n📘 Shannon Entropy: {entropy:.4f} bits")
    except ValueError:
        print("❌ Invalid probabilities. Must be numbers separated by commas.")
    except Exception as e:
        print(f"❌ Error: {e}")


def run_renyi_entropy() -> None:
    """Compute Renyi entropy."""
    print("\n📘 Renyi Entropy")
    print("H_α(X) = (1/(1-α)) log₂ ∑ p(x)^α")
    print("Enter probability distribution and order α.")
    print("Example: Probabilities: 0.5, 0.3, 0.2")
    print("         Order α: 2")
    
    prob_str = input("Probabilities: ").strip()
    alpha_str = input("Order α: ").strip()
    
    try:
        probabilities = [float(p.strip()) for p in prob_str.split(",")]
        alpha = float(alpha_str)
        entropy = compute_renyi_entropy(probabilities, alpha)
        print(f"\n📘 Renyi Entropy (α={alpha}): {entropy:.4f} bits")
    except ValueError:
        print("❌ Invalid input. Probabilities must be numbers, α must be a number.")
    except Exception as e:
        print(f"❌ Error: {e}")


def run_tsallis_entropy() -> None:
    """Compute Tsallis entropy."""
    print("\n📘 Tsallis Entropy")
    print("S_q(X) = (1/(q-1)) ∑ [p(x)^q - p(x)]")
    print("Enter probability distribution and parameter q.")
    print("Example: Probabilities: 0.5, 0.3, 0.2")
    print("         Parameter q: 2")
    
    prob_str = input("Probabilities: ").strip()
    q_str = input("Parameter q: ").strip()
    
    try:
        probabilities = [float(p.strip()) for p in prob_str.split(",")]
        q = float(q_str)
        entropy = compute_tsallis_entropy(probabilities, q)
        print(f"\n📘 Tsallis Entropy (q={q}): {entropy:.4f}")
    except ValueError:
        print("❌ Invalid input. Probabilities must be numbers, q must be a number.")
    except Exception as e:
        print(f"❌ Error: {e}")