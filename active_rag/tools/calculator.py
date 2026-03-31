"""Calculator tool for the LLM."""

import ast
import operator
import math


def safe_eval(expr: str) -> str:
    """Evaluate a mathematical expression safely."""
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.BitXor: operator.xor,
        ast.USub: operator.neg,
    }
    
    allowed_functions = {
        name: getattr(math, name) for name in dir(math) if not name.startswith("_")
    }

    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            if type(node.op) in allowed_operators:
                return allowed_operators[type(node.op)](left, right)
            raise ValueError(f"Operator {type(node.op).__name__} not allowed")
        elif isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            if type(node.op) in allowed_operators:
                return allowed_operators[type(node.op)](operand)
            raise ValueError(f"Unary operator {type(node.op).__name__} not allowed")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in allowed_functions:
                func = allowed_functions[node.func.id]
                args = [_eval(arg) for arg in node.args]
                return func(*args)
            raise ValueError(f"Function {node.func.id if isinstance(node.func, ast.Name) else 'unknown'} strictly prohibited.")
        else:
            raise TypeError(f"Unsupported expression node: {type(node).__name__}")

    try:
        node = ast.parse(expr, mode='eval').body
        result = _eval(node)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate mathematical expressions. Use this for all math.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate (e.g. '2 + 2 * 3' or 'math.sqrt(16)'). Do not use 'math.' prefix, just 'sqrt(16)'",
                }
            },
            "required": ["expression"],
        },
    }
}

def execute(kwargs: dict) -> str:
    return safe_eval(kwargs.get("expression", ""))
