# Contributing to OCEAN

We love your input! We want to make contributing to OCEAN as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Development Setup

1. Clone your fork:
   ```bash
   git clone https://github.com/your-username/ocean.git
   cd ocean
   ```

2. Set up development environment:
   ```bash
   make dev-setup
   ```
   This will:
   - Create a virtual environment
   - Install dependencies
   - Install pre-commit hooks
   - Create a `.env` file from template

3. Activate virtual environment:
   ```bash
   source .venv/bin/activate  # On Unix
   .venv\Scripts\activate     # On Windows
   ```

## Code Quality

We use several tools to ensure code quality:

- `black` for code formatting
- `isort` for import sorting
- `flake8` for style guide enforcement
- `mypy` for type checking
- `pytest` for testing

Run all checks with:
```bash
make lint
make test
```

Format code with:
```bash
make format
```

## Pull Request Process

1. Update the README.md with details of changes to the interface, if applicable.
2. Update the docs/ with any new documentation.
3. The PR will be merged once you have the sign-off of at least one maintainer.

## Any contributions you make will be under the MIT Software License
In short, when you submit code changes, your submissions are understood to be under the same [MIT License](http://choosealicense.com/licenses/mit/) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using GitHub's [issue tracker](https://github.com/yourusername/ocean/issues)
We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/yourusername/ocean/issues/new).

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can.
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## License
By contributing, you agree that your contributions will be licensed under its MIT License. 