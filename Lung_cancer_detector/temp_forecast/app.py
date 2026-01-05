from flask import Flask, request, jsonify, send_from_directory
import os
import joblib
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

app = Flask(__name__, static_folder='static', template_folder='.')

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
JSON_PATHS = [
    os.path.join(os.path.dirname(__file__), 'mecca_weather_hourly.json'),
    os.path.join(os.path.dirname(__file__), '..', '..', 'Downloads', 'mecca_weather_hourly.json')
]


def try_load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None
    return None


def detect_fields(records):
    # Try to find datetime-like and temperature-like fields
    if not records:
        return None, None
    sample = records[0]
    dt_key = None
    t_key = None
    dt_candidates = ['datetime', 'date', 'time', 'dt', 'timestamp', 'dt_iso']
    t_candidates = ['temp', 'temperature', 'air_temperature', 't', 'temp_c']
    for k in dt_candidates:
        if k in sample:
            dt_key = k
            break
    for k in t_candidates:
        if k in sample:
            t_key = k
            break
    return dt_key, t_key


def load_json_try(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception:
        return None


def prepare_series_from_json(path):
    data = load_json_try(path)
    if not data:
        return None
    dt_key, t_key = detect_fields(data)
    if not dt_key or not t_key:
        # Try to infer temperature and datetime by scanning keys
        # fallback: look for any numeric field and any string field
        keys = data[0].keys()
        for k in keys:
            if isinstance(data[0][k], (int, float)) and t_key is None:
                t_key = k
            if isinstance(data[0][k], str) and dt_key is None:
                dt_key = k
    if not dt_key or not t_key:
        return None
    rows = []
    for rec in data:
        try:
            dt_val = rec.get(dt_key)
            # handle unix timestamp
            if isinstance(dt_val, (int, float)):
                dt = datetime.utcfromtimestamp(int(dt_val))
            else:
                dt = datetime.fromisoformat(dt_val)
            temp = rec.get(t_key)
            temp = float(temp)
            rows.append((dt, temp))
        except Exception:
            continue
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=['dt', 'temp']).sort_values('dt').drop_duplicates('dt')
    df = df.set_index('dt').asfreq('H').interpolate()
    return df


def train_simple_model_from_df(df):
    # Build lag features (previous 24 hours) to predict next hour
    X = []
    y = []
    temps = df['temp'].values
    if len(temps) < 50:
        return None
    for i in range(24, len(temps)-1):
        X.append(temps[i-24:i])
        y.append(temps[i])
    X = np.array(X)
    y = np.array(y)
    from sklearn.ensemble import RandomForestRegressor
    m = RandomForestRegressor(n_estimators=50, random_state=0)
    m.fit(X, y)
    try:
        joblib.dump(m, MODEL_PATH)
    except Exception:
        pass
    return m


def get_or_train_model():
    m = try_load_model()
    if m is not None:
        return m, None
    # try to find json data to train
    for p in JSON_PATHS:
        df = prepare_series_from_json(p)
        if df is not None:
            m = train_simple_model_from_df(df)
            if m is not None:
                return m, df
    return None, None


def predict_next_hours(start_dt, model, df_for_init=None, hours=72):
    # Build initial last 24 hours array from df_for_init if available
    last_vals = None
    if df_for_init is not None:
        # find data up to start_dt
        try:
            series = df_for_init['temp']
            if start_dt in series.index:
                idx = series.index.get_loc(start_dt)
                start_pos = idx
            else:
                # find previous timestamp
                prev_idx = series.index.asof(start_dt)
                if pd.isna(prev_idx):
                    prev_idx = series.index[0]
                start_pos = series.index.get_loc(prev_idx)
            arr = series.values
            if start_pos >= 24:
                last_vals = list(arr[start_pos-24:start_pos])
            else:
                # pad with mean
                pad = [float(np.nanmean(arr[:start_pos+1]))] * (24 - start_pos)
                last_vals = pad + list(arr[:start_pos])
        except Exception:
            last_vals = None
    if last_vals is None:
        # default: zeros
        last_vals = [0.0] * 24
    preds = []
    cur = last_vals.copy()
    for i in range(hours):
        x = np.array(cur[-24:]).reshape(1, -1)
        try:
            p = float(model.predict(x)[0])
        except Exception:
            p = float(np.mean(cur))
        preds.append(p)
        cur.append(p)
    timestamps = [start_dt + timedelta(hours=i) for i in range(hours)]
    return list(zip(timestamps, preds))


@app.route('/')
def index():
    return send_from_directory(os.path.dirname(__file__), 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json() or {}
    date_str = payload.get('start_date')
    if not date_str:
        return jsonify({'error': 'start_date required (YYYY-MM-DD or ISO)'}), 400
    try:
        # try various formats
        try:
            start_dt = datetime.fromisoformat(date_str)
        except Exception:
            start_dt = datetime.strptime(date_str, '%Y-%m-%d')
    except Exception:
        return jsonify({'error': 'invalid date format'}), 400

    model, df = get_or_train_model()
    if model is None:
        # fallback naive cyclical daily pattern
        hours = 72
        res = []
        for i in range(hours):
            dt = start_dt + timedelta(hours=i)
            # basic sinusoidal daily temps around 25C with noise
            val = 25 + 7 * np.sin(2 * np.pi * (dt.hour / 24.0))
            res.append({'dt': dt.isoformat(), 'temp': round(float(val), 2)})
        return jsonify({'method': 'fallback', 'predictions': res})

    preds = predict_next_hours(start_dt, model, df_for_init=df, hours=72)
    out = [{'dt': t.isoformat(), 'temp': round(float(v), 2)} for t, v in preds]
    return jsonify({'method': 'model', 'predictions': out})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
