# Lab 1: Calculator Testing and GitHub Actions

Based on [MLOps Lab 1](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Github_Labs/Lab1).

## My Changes
- Added `divide(x, y)` with division-by-zero handling.
- Added `power(x, y)` with negative and fractional exponent support.
- Added pytest and unittest coverage for both functions and invalid inputs.

## Setup and Test
Run from the repository root:

```bash
cd lab1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest test/test_pytest.py -v
python -m unittest test.test_unittest -v