# Agent Instructions for Pro-Trader TA API

## System Dependencies
- This project relies on the **TA-Lib C-library**. If you add new dependencies, ensure they are compatible with the `python:3.10-slim` base image used in the `Dockerfile`.

## Code Organization
- `main.py`: Contains all logic. For scaling, consider moving indicators to a `lib/` directory.
- `test_api.py`: Comprehensive test suite. Always run this before submitting changes.

## Adding New Indicators
1. Add the indicator logic to `get_indicator_results_sync`.
2. Update `INDICATOR_METADATA` so the indicator appears in `GET /indicators`.
3. If it's a "custom" indicator (non-standard), add it to the `res` dictionary in its own category.

## Performance
- Keep `get_indicator_results_sync` efficient. It runs in a thread pool to avoid blocking the event loop.
- Use `np.vectorize` or Pandas/Numpy operations instead of loops whenever possible.
