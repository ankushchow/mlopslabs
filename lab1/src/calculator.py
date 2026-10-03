def fun1(x, y):
    """Add two numbers."""
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        raise ValueError("Both inputs must be numbers.")
    return x + y


def fun2(x, y):
    """Subtract y from x."""
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        raise ValueError("Both inputs must be numbers.")
    return x - y


def fun3(x, y):
    """Multiply two numbers."""
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        raise ValueError("Both inputs must be numbers.")
    return x * y


def fun4(x, y, z):
    """Add three results together."""
    return x + y + z


def divide(x, y):
    """Divide x by y, rejecting division by zero."""
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        raise ValueError("Both inputs must be numbers.")
    if y == 0:
        raise ValueError("Cannot divide by zero.")
    return x / y


def power(x, y):
    """Raise x to the power y."""
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        raise ValueError("Both inputs must be numbers.")
    if x == 0 and y < 0:
        raise ValueError("Zero cannot be raised to a negative power.")
    return x ** y