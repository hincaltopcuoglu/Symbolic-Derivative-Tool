# 📐 Symbolic Derivative Tool

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A Python-based symbolic differentiation and analysis tool using Python's built-in AST. It supports:

- ✅ Symbolic Derivatives
- ✅ Gradient and Hessian computation
- ✅ Limit-based and numerical approximation (Forward & Central Difference)
- ✅ Laplacian Operator
- ✅ Taylor Series Expansion (multivariate)
- ✅ Directional Derivatives
- ✅ Symbolic Chain Rule Expansion
- ✅ Jacobian Matrix for multivariable functions

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
python derivative_tool.py
```
You'll be prompted to enter a function and then choose from a list of operations.

---

## 🧪 Usage Examples

### Example 1: Univariate Function
```text
Enter a function: f(x) = sin(x**2 + 3*x)
Differentiate with respect to: x
Enter values for variables (e.g. x=2): x=1
```

### Example 2: Multivariate Function
```text
Enter a function: f(x, y) = x**2 + y**2
Differentiate with respect to: x
Enter values for variables (e.g. x=1, y=2): x=1, y=2
```

---

## 📘 Features Menu

After entering the function and point, you'll see:

```text
🧪 Select an operation to perform:
1. Compare Derivative Approximations (Limit-based)
2. Compute Gradient
3. Compare Gradient with Numerical Derivatives
4. Compute Hessian Matrix
5. Compute Laplacian
6. Taylor Series
7. Compute Directional Derivative
8. Symbolic Chain Rule Expansion
9. Exit
```

Choose a number to run that operation.

---

## 📁 Project Structure

```
Symbolic-Derivative-Tool/
├── derivative_tool.py
├── README.md
├── .gitignore
└── requirements.txt
```

---

## 🧠 Author

Built with ❤️ and symbolic brainpower by Hincal Topcuoglu.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
