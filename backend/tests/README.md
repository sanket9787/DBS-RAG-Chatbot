# Backend Test Suite

## Running Tests

### Run all tests
```bash
cd backend
pytest
```

### Run specific test categories
```bash
# Unit tests only
pytest tests/unit -m unit

# Integration tests only
pytest tests/integration -m integration

# Performance tests only
pytest tests/performance -m performance

# Skip slow tests
pytest -m "not slow"
```

### Run with coverage
```bash
pytest --cov=services --cov=api --cov-report=html
```

### Run specific test file
```bash
pytest tests/unit/test_query_processor.py
```

### Run specific test
```bash
pytest tests/unit/test_query_processor.py::TestQueryProcessor::test_normalize_query
```

## Test Structure

- `tests/unit/` - Unit tests for individual components
- `tests/integration/` - Integration tests for API endpoints and component interactions
- `tests/performance/` - Performance benchmarks and load tests
- `conftest.py` - Shared fixtures and test configuration

## Writing New Tests

1. Create test file in appropriate directory
2. Import pytest and necessary modules
3. Use fixtures from `conftest.py` when possible
4. Follow naming convention: `test_*.py` for files, `test_*` for functions
5. Add appropriate markers (`@pytest.mark.unit`, `@pytest.mark.integration`, etc.)

## Coverage Goals

- Target: 70%+ code coverage
- Critical paths: 90%+ coverage
- Focus areas: RAG service, query processing, API routes

