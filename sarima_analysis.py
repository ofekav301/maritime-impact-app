import pandas as pd
import pmdarima as pm
from statsmodels.tsa.stattools import acf
from scipy.signal import find_peaks

def infer_seasonality(train_series, max_m):
    """Automatically infers the dominant seasonal period using differenced ACF and peak finding."""
    if len(train_series) < max_m * 2:
        return 1 
        
    diff_series = train_series.diff().dropna()
    if diff_series.empty:
        return 1
        
    acf_values = acf(diff_series, nlags=max_m, fft=True)
    acf_values[0:2] = 0  
    
    peaks, _ = find_peaks(acf_values, prominence=0.15)
    
    if len(peaks) > 0:
        dominant_period = int(max(peaks, key=lambda p: acf_values[p]))
        if 2 < dominant_period <= max_m:
            return dominant_period
            
    return 1

def run_sarima_impact_analysis(df, feature, event_date, pre_months, post_days, resolution):
    """Resamples data, infers seasonality, and trains Auto-SARIMA."""
    data = df[[feature]].copy()
    event_dt = pd.to_datetime(event_date)
    
    if resolution == 'Weekly':
        data = data.resample('W').sum()
        max_search_m = 53 
    elif resolution == 'Monthly':
        data = data.resample('ME').sum()
        max_search_m = 25 
    else: 
        data = data.resample('D').sum()
        max_search_m = 35 
        
    start_pre = event_dt - pd.DateOffset(months=pre_months)
    end_post = event_dt + pd.Timedelta(days=post_days)
    
    data = data[(data.index >= start_pre) & (data.index <= end_post)]
    if data.empty:
        raise ValueError("Not enough data in the selected date range.")

    train = data[data.index < event_dt][feature]
    test = data[data.index >= event_dt][feature]
    
    if len(train) < 10:
        raise ValueError(f"Not enough historical data ({len(train)} points). Increase 'Months of history'.")
    if len(test) == 0:
        raise ValueError("No post-event data available.")

    inferred_m = infer_seasonality(train, min(max_search_m, len(train) // 2))
    seasonal = inferred_m > 1

    model = pm.auto_arima(
        train, 
        seasonal=seasonal, 
        m=inferred_m if seasonal else 1,
        suppress_warnings=True, 
        error_action='ignore',
        trace=False
    )
    
    forecast, conf_int = model.predict(n_periods=len(test), return_conf_int=True, alpha=0.05)
    
    results = {
        'train': train,
        'test': test,
        'forecast': pd.Series(forecast, index=test.index),
        'conf_lower': pd.Series(conf_int[:, 0], index=test.index),
        'conf_upper': pd.Series(conf_int[:, 1], index=test.index),
        'model_summary': str(model.summary()),
        'inferred_m': inferred_m
    }
    return results
