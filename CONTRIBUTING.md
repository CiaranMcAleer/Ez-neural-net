# Contributing to Ez Vision

Thank you for your interest in contributing to Ez Vision! This guide will help you get started.

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/ez-vision.git
   cd ez-vision
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e .[dev]  # Install development dependencies
   ```

4. **Run tests to verify setup:**
   ```bash
   python tests.py
   ```

## How to Contribute

### Reporting Bugs
- Use the GitHub issue tracker
- Include Python version, OS, and error messages
- Provide a minimal example that reproduces the issue

### Suggesting Features
- Open an issue with the "enhancement" label
- Describe the use case and expected behavior
- Consider backward compatibility

### Code Contributions

1. **Fork the repository**
2. **Create a feature branch:**
   ```bash
   git checkout -b feature-amazing-feature
   ```

3. **Make your changes:**
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

4. **Run tests:**
   ```bash
   python tests.py
   ```

5. **Commit and push:**
   ```bash
   git commit -m "Add amazing feature"
   git push origin feature-amazing-feature
   ```

6. **Open a Pull Request**

## Code Style

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings for all public functions and classes
- Keep functions focused and small

## Testing

- All new features must include tests
- Tests should be robust and use real data (no mocks)
- Maintain the existing test coverage

## Documentation

- Update README.md for user-facing changes
- Add examples for new features
- Update CHANGELOG.md with your changes

## Questions?

Feel free to open an issue or reach out to the maintainers!
