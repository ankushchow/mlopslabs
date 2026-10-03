import pytest

from src.calculator import fun1, fun2, fun3, fun4, divide, power


def test_add():
    assert fun1(2, 3) == 5


def test_subtract():
    assert fun2(2, 3) == -1


def test_multiply():
    assert fun3(2, 3) == 6


def test_combined_results():
    assert fun4(fun1(2, 3), fun2(2, 3), fun3(2, 3)) == 10


@pytest.mark.parametrize(
    "x, y, expected",
    [
        (10, 2, 5),
        (7, 2, 3.5),
        (-9, 3, -3),
        (0, 5, 0),
    ],
)
def test_divide(x, y, expected):
    assert divide(x, y) == pytest.approx(expected)


def test_divide_by_zero():
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        divide(10, 0)


@pytest.mark.parametrize(
    "x, y, expected",
    [
        (2, 3, 8),
        (5, 0, 1),
        (2, -2, 0.25),
        (-2, 3, -8),
        (9, 0.5, 3),
        (0, 3, 0),
    ],
)
def test_power(x, y, expected):
    assert power(x, y) == pytest.approx(expected)


def test_zero_to_negative_power():
    with pytest.raises(ValueError, match="Zero cannot be raised"):
        power(0, -1)


@pytest.mark.parametrize("operation", [fun1, fun2, fun3, divide, power])
@pytest.mark.parametrize("x, y", [("hello", 2), (2, None)])
def test_invalid_inputs(operation, x, y):
    with pytest.raises(ValueError, match="Both inputs must be numbers"):
        operation(x, y)