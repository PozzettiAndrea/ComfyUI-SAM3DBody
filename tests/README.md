# ComfyUI-SAM3DBody Tests

This directory contains unit and integration tests for ComfyUI-SAM3DBody.

## Test Structure

- `conftest.py` - Pytest configuration and ComfyUI mocking
- `test_node_registration.py` - Unit tests for node registration (fast)
- `test_inference.py` - Integration tests with real models (slow)

## Running Tests

### Install Test Dependencies

```bash
pip install pytest pytest-cov
```

### Run All Tests

```bash
pytest tests/ -v
```

### Run Only Unit Tests (Fast)

```bash
pytest tests/ -v -m unit
```

### Run Only Integration Tests (Slow)

Requires HuggingFace token for model downloads:

```bash
export HF_TOKEN=your_token_here
pytest tests/ -v -m integration
```

Or run specific integration tests:

```bash
pytest tests/test_inference.py -v -s -m integration
```

### Run Tests with Coverage

```bash
pytest tests/ -v --cov=nodes --cov=sam_3d_body --cov-report=html
```

Then open `htmlcov/index.html` to view the coverage report.

## Test Markers

Tests are organized using pytest markers:

- `@pytest.mark.unit` - Fast tests, no model loading required
- `@pytest.mark.integration` - Slow tests with real model inference

## CI/CD

Tests run automatically on push/PR via GitHub Actions:

- **test-install**: Runs unit tests on Ubuntu, Windows, macOS
- **test-cpu-inference**: Runs integration tests with CPU inference

See `.github/workflows/test-install.yml` for details.

## Adding New Tests

When adding new nodes or features:

1. Add unit tests to `test_node_registration.py` for structure validation
2. Add integration tests to `test_inference.py` for actual functionality
3. Use appropriate markers (`@pytest.mark.unit` or `@pytest.mark.integration`)
4. Update this README if needed

## Troubleshooting

### ComfyUI Import Errors

The `conftest.py` file mocks all ComfyUI dependencies. If you see import errors, ensure:

1. Tests are run from the repository root
2. The conftest.py is properly mocking ComfyUI modules
3. Node imports are direct (not through `__init__.py`)

### Integration Test Failures

Integration tests require:

1. HuggingFace token (`HF_TOKEN` environment variable)
2. Internet connection for model downloads
3. Sufficient disk space (~5GB for model cache)
4. Time (first run downloads models, subsequent runs use cache)

### Skipping __init__.py

The `__init__.py` file has pytest detection and will skip loading nodes during tests. To force initialization (not recommended):

```bash
SAM3DB_FORCE_INIT=1 pytest tests/
```
