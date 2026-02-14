# Agent Instructions for Pro-Trader TA API

## System Dependencies
- This project relies on the **TA-Lib C-library**. The Dockerfile handles this installation correctly.
- Ensure all new dependencies are listed in `requirements.txt`.

## Code Organization
- **Separation of Concerns**: Routing is in `main.py`, while all mathematical and indicator logic is in `lib/indicators.py`.
- **Database/Auth**: Persistence logic is in `lib/database.py` and authentication/monetization in `lib/auth.py`.

## Adding New Indicators
1. Implement the calculation logic in `lib/indicators.py`.
2. Add the indicator to the relevant category in the `_get_indicators_metadata()` function.
3. Integrate the call into `get_indicator_results_sync()`.
4. Ensure `NaN` and `Inf` values are handled using the `clean_dict` utility before returning JSON.

## Performance & Scaling
- CPU-bound calculations are parallelized via `ProcessPoolExecutor` in `lib/indicators.py`.
- Large responses are compressed using `GZipMiddleware`.
- Use vectorized NumPy/Pandas operations. Avoid large loops in indicator logic.
- External data requests are cached for 60 seconds using `diskcache`.

## Verification
- Use `test_api.py` to verify all core endpoints.
- Ensure the interactive chart (`/analyze/chart`) renders correctly after adding new visual overlays.
