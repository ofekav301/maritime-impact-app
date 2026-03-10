import pandas as pd
from prophet import Prophet
import logging

# Suppress Prophet's verbose console logs
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

def run_prophet_impact_analysis(df, feature, event_date, pre_months, post_days, resolution):
    """Trains a Prophet Additive Model on pre-event data and forecasts post-event."""
    data = df[[feature]].copy()
    event_dt = pd.to_datetime(event_date)
    
    if resolution == 'Weekly':
        data = data.resample('W').sum()
    elif resolution == 'Monthly':
        data = data.resample('ME').sum()
    else: 
        data = data.resample('D').sum()
        
    start_pre = event_dt - pd.DateOffset(months=pre_months)
    end_post = event_dt + pd.Timedelta(days=post_days)
    
    data = data[(data.index >= start_pre) & (data.index <= end_post)]
    if data.empty:
        raise ValueError("Not enough data in the selected date range.")

    train = data[data.index < event_dt]
    test = data[data.index >= event_dt]
    
    if len(train) < 10:
        raise ValueError(f"Not enough historical data ({len(train)} points). Increase 'Months of history'.")
    if len(test) == 0:
        raise ValueError("No post-event data available.")

    df_train = pd.DataFrame({'ds': train.index, 'y': train[feature]})
    df_test = pd.DataFrame({'ds': test.index})

    model = Prophet(
        yearly_seasonality=True if pre_months >= 12 else 'auto',
        weekly_seasonality=True if resolution in ['Daily'] else False,
        daily_seasonality=False,
        interval_width=0.95 
    )
    model.fit(df_train)
    
    forecast_df = model.predict(df_test)
    
    results = {
        'train': train[feature],
        'test': test[feature],
        'forecast': pd.Series(forecast_df['yhat'].values, index=test.index),
        'conf_lower': pd.Series(forecast_df['yhat_lower'].values, index=test.index),
        'conf_upper': pd.Series(forecast_df['yhat_upper'].values, index=test.index),
        'model_summary': "Prophet Additive Model (Trend + Fourier Seasonality components)",
        'inferred_m': "Prophet"
    }
    return results
