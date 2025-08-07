To get pytest to print the total coverage for your tests, you'll need to use the pytest-cov plugin. Here's how you can set it up and run your tests with coverage reporting:

1. Run your tests with coverage:

```
pytest --cov=src --cov-report=term-missing
```

This command will run your tests and show the coverage report in the terminal. The `--cov=src` option specifies the directory to measure coverage for (adjust 'src' to match your project structure if needed). The `--cov-report=term-missing` option will show which lines are missing coverage.

2. If you want to see the total coverage percentage at the end of the report, you can add the `--cov-report=term` option:

```
pytest --cov=src --cov-report=term-missing --cov-report=term
```

This will display a summary at the end of the report with the total coverage percentage.

3. If you want to generate an HTML report for more detailed coverage information, you can use:

```
pytest --cov=src --cov-report=html
```

This will create a directory named `htmlcov` with an interactive HTML report of your coverage.
