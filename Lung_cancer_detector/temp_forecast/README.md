# Simple Temperature Forecast Web App

This small app provides hourly temperature forecasts for the 3 days (72 hours) after a given start date.

How it works
- If a trained `model.pkl` exists in the same folder, the server will load it and use it.
- Else, if `mecca_weather_hourly.json` is placed in the same folder (or in your Downloads), the server will attempt to train a simple model from that data and save `model.pkl`.
- Otherwise the server uses a simple fallback daily-cycle heuristic.

Run locally

1. Create a Python environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. (Optional) Put your trained `model.pkl` next to `app.py`, or put `mecca_weather_hourly.json` in the same folder to allow on-start training.

3. Start the server:

```bash
python app.py
```

4. Open http://127.0.0.1:5000/ in your browser.

Notes
- The app expects the JSON to contain a datetime-like field (e.g., `datetime`, `date`, `dt`) and a temperature field (e.g., `temp`, `temperature`).
- If you want, I can adapt the loader to your exact JSON schema or wire your already-trained model into this app.
