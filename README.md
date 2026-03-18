# 📐 Symbolic Derivative Tool

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A Python-based symbolic differentiation tool using Python's built-in AST. It computes symbolic derivatives of mathematical expressions and supports optimization with constraints using the Lagrange multiplier method, entropy calculations, and Jackson q-derivatives.

- ✅ Pure Symbolic Differentiation
- ✅ Support for basic arithmetic operations (+, -, *, /, **)
- ✅ Trigonometric functions (sin, cos, etc.)
- ✅ Exponential and logarithmic functions
- ✅ Chain rule, product rule, quotient rule, power rule
- ✅ Pretty-printed output with superscripts
- ✅ **Lagrange Multiplier Method** for constrained optimization
- ✅ **Shannon, Renyi, and Tsallis Entropy** calculations
- ✅ **Jackson q-Derivative** for quantum calculus

---

## 🚀 Getting Started

### 🔧 Installation

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/Symbolic-Derivative-Tool.git
cd Symbolic-Derivative-Tool
```

2. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

3. **🚀 Usage**
```bash
python main.py
```
You'll be prompted to choose between symbolic differentiation, Lagrange multiplier method, and advanced mathematical computations.

---

## 🧪 Usage Examples

### Example 1: Simple polynomial derivative
```text
Choose an operation:
1. Symbolic Differentiation
2. Lagrange Multiplier Method
3. Advanced Mathematical Computations
> 1

Choose differentiation type:
1. Standard derivative
2. Jackson q-derivative
> 1

Function: f(x) = x**2 + 3*x
Variable to differentiate with respect to: x
📘 Symbolic derivative: 2x + 3
```

### Example 2: Jackson q-Derivative
```text
Choose differentiation type:
1. Standard derivative
2. Jackson q-derivative
> 2

Function: f(x) = x**2
q parameter: 2
Variable to differentiate with respect to: x
📘 Jackson q-derivative (q=2): 3x
```

### Example 3: Lagrange Multiplier Method
```text
Choose an operation:
1. Symbolic Differentiation
2. Lagrange Multiplier Method
3. Advanced Mathematical Computations
> 2

Objective function f: x**2 + y**2
Constraint function g: x + y
📘 Variables: x, y

📘 Lagrange Equations:
∇f = λ ∇g
∂f/∂x = λ ∂g/∂x
2x = λ 1
∂f/∂y = λ ∂g/∂y
2y = λ 1

📘 Constraint:
g(x,y,...) = c
x + y = c
```

### Example 4: Shannon Entropy
```text
Choose an operation:
1. Symbolic Differentiation
2. Lagrange Multiplier Method
3. Advanced Mathematical Computations
> 3

Choose computation:
1. Shannon Entropy
2. Renyi Entropy
3. Tsallis Entropy
> 1

Probabilities: 0.5, 0.3, 0.2
📘 Shannon Entropy: 1.4855 bits
```
```text
Function: sin(x)
Variable to differentiate with respect to: x
📘 Symbolic derivative: cosx
```

---

## 🧪 Testing

Run the unit tests to validate the functionality:

```bash
python -m pytest tests/ -v
```

---

## 📁 Project Structure

```
Symbolic-Derivative-Tool/
├── main.py                 # Entry point
├── cli.py                  # Command-line interface with menu options
├── differentiator.py       # Core differentiation logic, gradient computation, and Jackson derivatives
├── entropy.py              # Entropy calculation functions (Shannon, Renyi, Tsallis)
├── parser.py               # Function parsing utilities
├── tests/                  # Unit tests (local only, not in repository)
│   ├── test_parser.py
│   ├── test_differentiator.py
│   ├── test_entropy.py
│   └── test_cli.py
├── README.md
├── requirements.txt
└── LICENCE
```

---

## 🧠 Author

Built with ❤️ and symbolic brainpower by Hincal Topcuoglu.

---

## 📄 License

This project is licensed under the [MIT License](LICENCE).
