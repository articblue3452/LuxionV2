def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b != 0:
        return a / b
    else:
        raise ValueError("Cannot divide by zero")

def calculate(operation, num1, num2):
    if operation == "add":
        return add(num1, num2)
    elif operation == "subtract":
        return subtract(num1, num2)
    elif operation == "multiply":
        return multiply(num1, num2)
    elif operation == "divide":
        return divide(num1, num2)
    else:
        raise ValueError(f"Invalid operation: {operation}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("Usage: python calculator.py <operation> <num1> <num2>")
        print("Supported operations: add, subtract, multiply, divide")
        sys.exit(1)

    operation = sys.argv[1]
    num1 = float(sys.argv[2])
    num2 = float(sys.argv[3])

    try:
        result = calculate(operation, num1, num2)
        print(f"Result: {result}")
    except ValueError as e:
        print(e)