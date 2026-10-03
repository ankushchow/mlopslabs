import unittest

from src.calculator import fun1, fun2, fun3, fun4, divide, power


class TestCalculator(unittest.TestCase):

    def test_add(self):
        self.assertEqual(fun1(2, 3), 5)

    def test_subtract(self):
        self.assertEqual(fun2(2, 3), -1)

    def test_multiply(self):
        self.assertEqual(fun3(2, 3), 6)

    def test_combined_results(self):
        result = fun4(fun1(2, 3), fun2(2, 3), fun3(2, 3))
        self.assertEqual(result, 10)

    def test_divide(self):
        cases = [
            (10, 2, 5),
            (7, 2, 3.5),
            (-9, 3, -3),
            (0, 5, 0),
        ]
        for x, y, expected in cases:
            with self.subTest(x=x, y=y):
                self.assertAlmostEqual(divide(x, y), expected)

    def test_divide_by_zero(self):
        with self.assertRaisesRegex(ValueError, "Cannot divide by zero"):
            divide(10, 0)

    def test_power(self):
        cases = [
            (2, 3, 8),
            (5, 0, 1),
            (2, -2, 0.25),
            (-2, 3, -8),
            (9, 0.5, 3),
            (0, 3, 0),
        ]
        for x, y, expected in cases:
            with self.subTest(x=x, y=y):
                self.assertAlmostEqual(power(x, y), expected)

    def test_zero_to_negative_power(self):
        with self.assertRaisesRegex(ValueError, "Zero cannot be raised"):
            power(0, -1)

    def test_invalid_inputs(self):
        for operation in [fun1, fun2, fun3, divide, power]:
            for x, y in [("hello", 2), (2, None)]:
                with self.subTest(operation=operation.__name__, x=x, y=y):
                    with self.assertRaisesRegex(
                        ValueError, "Both inputs must be numbers"
                    ):
                        operation(x, y)


if __name__ == "__main__":
    unittest.main()