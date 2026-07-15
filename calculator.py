import math

def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

def square_root(a):
    return math.sqrt(a)

def factorial(n):
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

if __name__ == "__main__":
    print("Calculator Demo")

    # Addition
    print("\nAddition:")
    print(add(5, 3))

    # Subtraction
    print("\nSubtraction:")
    print(subtract(10, 7))

    # Multiplication
    print("\nMultiplication:")
    print(multiply(4, 6))

    # Division
    print("\nDivision:")
    print(divide(15, 5))

    # Square Root
    print("\nSquare Root:")
    print(square_root(16))

    # Factorial
    print("\nFactorial:")
    print(factorial(5))