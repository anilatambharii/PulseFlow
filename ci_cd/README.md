# CI/CD Directory

This directory contains CI/CD pipeline configurations and tests for the Enterprise MLOps project.

## Structure

ci_cd/
├── github_actions.yml # GitHub Actions workflow (place in .github/workflows/)
├── tests/
│ ├── test_app.py # API and integration tests
│ └── test_etl.py # ETL pipeline tests
├── pytest.ini # Pytest configuration
└── README.md # This file

text

## GitHub Actions Workflow

The CI/CD pipeline includes:

1. **Lint and Test**: Code quality checks and unit tests
2. **ETL Pipeline**: Run and validate data processing
3. **Model Training**: Train and validate model
4. **API Testing**: Test FastAPI endpoints
5. **Docker Build**: Build and test Docker images
6. **Deploy**: Deployment step (production only)

## Running Tests Locally

### Install test dependencies
pip install pytest pytest-cov flake8 black

text

### Run all tests
pytest ci_cd/tests/ -v

text

### Run specific test file
pytest ci_cd/tests/test_app.py -v

text

### Run with coverage
pytest ci_cd/tests/ -v --cov=. --cov-report=html

text

### Check code formatting
black --check .

text

### Lint code
flake8 .

text

## Test Categories

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **API Tests**: Test FastAPI endpoints
- **ETL Tests**: Test data pipeline

## Adding New Tests

1. Create test file in `ci_cd/tests/` with `test_` prefix
2. Follow pytest naming conventions
3. Use appropriate markers (@pytest.mark.unit, @pytest.mark.integration)
4. Run tests locally before committing

## Continuous Integration

Push to `main` or `develop` branches triggers:
- Automated testing
- Code quality checks
- Model training validation
- Docker image building
- Deployment (main branch only)
How to Run Tests Locally
1. Install test dependencies:
bash
pip install pytest pytest-cov flake8 black
2. Generate model (required for tests):
bash
python models/generate_model.py
3. Run all tests:
bash
pytest ci_cd/tests/ -v
4. Run with coverage:
bash
pytest ci_cd/tests/ -v --cov=. --cov-report=html
Output:

text
ci_cd/tests/test_app.py::TestAPIEndpoints::test_root_endpoint PASSED
ci_cd/tests/test_app.py::TestAPIEndpoints::test_health_endpoint PASSED
ci_cd/tests/test_app.py::TestAPIEndpoints::test_predict_endpoint PASSED
...
====== 15 passed in 3.42s ======
This CI/CD module provides complete automated testing and continuous integration for your Enterprise MLOps Pipeline, ensuring code quality and system reliability with every commit.