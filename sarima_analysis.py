import pandas as pd
import numpy as np
import pmdarima as pm
import plotly.graph_objects as go
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks

def infer_seasonality(train_series, max_m):
    """Automatically infers the dominant seasonal period using differenced ACF and peak finding."""
    if len(train_series) < max_m * 2:
        return 1 
        
    # 1. Take the first difference to expose the pure seasonal heartbeat
    diff_series = train_series.diff().dropna()
    if diff_series.empty:
        return 1
        
    # 2. Calculate ACF on the stationary differenced data
    acf_values = acf(diff_series, nlags=max_m, fft=True)
    acf_values[0:2] = 0  # Ignore immediate lag noise
    
    # 3. Find true rhythmic peaks, ignoring random background noise
    peaks, _ = find_peaks(acf_values, prominence=0.15)
    
    if len(peaks) > 0:
        # Out of all the rhythmic peaks found, grab the one with the strongest correlation
        dominant_period = int(max(peaks, key=lambda p: acf_values[p]))
        
        # Ensure it's a valid cycle length (> 2) and within our search window
        if 2 < dominant_period <= max_m:
            return dominant_period
            
    return 1

def run_sarima_impact_analysis(df, feature, event_date, pre_months, post_days, resolution):
    """Resamples data, infers seasonality, and trains Auto-SARIMA."""
    data = df[[feature]].copy()
    event_dt = pd.to_datetime(event_date)
    
    # 1. Resample based on user resolution & set max search distance for seasonality
    if resolution == 'Weekly':
        data = data.resample('W').sum()
        max_search_m = 53 # Look for up to a 1-year cycle
    elif resolution == 'Monthly':
        data = data.resample('ME').sum()
        max_search_m = 25 # Look for up to a 2-year cycle
    else: # Daily
        data = data.resample('D').sum()
        max_search_m = 35 # Look for up to a monthly cycle in daily data
        
    # 2. Calculate time windows
    start_pre = event_dt - pd.DateOffset(months=pre_months)
    end_post = event_dt + pd.Timedelta(days=post_days)
    
    # Filter dataset
    data = data[(data.index >= start_pre) & (data.index <= end_post)]
    if data.empty:
        raise ValueError("Not enough data in the selected date range.")

    # 3. Split into Train and Test
    train = data[data.index < event_dt][feature]
    test = data[data.index >= event_dt][feature]
    
    if len(train) < 10:
        raise ValueError(f"Not enough historical data ({len(train)} points). Increase 'Months of history'.")
    if len(test) == 0:
        raise ValueError("No post-event data available.")

    # 4. Automatically Infer Seasonality
    inferred_m = infer_seasonality(train, min(max_search_m, len(train) // 2))
    seasonal = inferred_m > 1

    # 5. Auto-SARIMA Training
    model = pm.auto_arima(
        train, 
        seasonal=seasonal, 
        m=inferred_m if seasonal else 1,
        suppress_warnings=True, 
        error_action='ignore',
        trace=False
    )
    
    # 6. Forecast
    forecast, conf_int = model.predict(n_periods=len(test), return_conf_int=True, alpha=0.05)
    
    results = {
        'train': train,
        'test': test,
        'forecast': pd.Series(forecast, index=test.index),
        'conf_lower': pd.Series(conf_int[:, 0], index=test.index),
        'conf_upper': pd.Series(conf_int[:, 1], index=test.index),
        'model_summary': str(model.summary()),
        'inferred_m': inferred_m # We pass this out to show the user!
    }
    return results

def plot_sarima_dashboard(results, feature, event_date, country, port):
    """Generates a connected Plotly chart comparing Actuals to SARIMA Forecast."""
    
    train = results['train']
    test = results['test']
    forecast = results['forecast']
    
    fig = go.Figure()

    # 1. Plot Actual Historical Data
    fig.add_trace(go.Scatter(
        x=train.index, y=train.values, 
        name='Historical Data (Pre-Event)', line=dict(color='#1f77b4', width=2)
    ))
    
    # --- FIX THE GAP ---
    anchor_date = [train.index[-1]]
    anchor_val = [train.iloc[-1]]
    
    test_x = anchor_date + test.index.tolist()
    test_y = anchor_val + test.values.tolist()
    
    forecast_x = anchor_date + forecast.index.tolist()
    forecast_y = anchor_val + forecast.values.tolist()
    
    conf_lower = anchor_val + results['conf_lower'].tolist()
    conf_upper = anchor_val + results['conf_upper'].tolist()
    # -------------------

    # 2. Plot Actual Post-Event Data (CHANGED COLOR TO BLACK)
    fig.add_trace(go.Scatter(
        x=test_x, y=test_y, 
        name='Actual Data (Post-Event)', line=dict(color='#000000', width=2.5)
    ))

    # 3. Plot SARIMA Forecast
    fig.add_trace(go.Scatter(
        x=forecast_x, y=forecast_y, 
        name='Expected (SARIMA Forecast)', line=dict(color='#d62728', width=2, dash='dash')
    ))

    # 4. Confidence Interval Shading
    fig.add_trace(go.Scatter(
        x=forecast_x + forecast_x[::-1],
        y=conf_upper + conf_lower[::-1],
        fill='toself', fillcolor='rgba(214, 39, 40, 0.15)', line=dict(color='rgba(255,255,255,0)'),
        name='Expected Range (95% CI)', showlegend=False
    ))

    # 5. Add Vertical Event Line
    event_date_str = pd.to_datetime(event_date).strftime('%Y-%m-%d')
    fig.add_vline(x=event_date_str, line_width=2, line_dash="dash", line_color="gray")
    
    fig.add_annotation(
        x=event_date_str, y=1.02, yref="paper", 
        text="Event Occurred", showarrow=False, xanchor="left",
        font=dict(color="black", size=12)
    )

    # 6. Lock the Y-Axis
    max_val = max(train.max(), test.max(), forecast.max())
    min_val = min(train.min(), test.min(), forecast.min())
    padding = (max_val - min_val) * 0.15 if (max_val - min_val) != 0 else max_val * 0.15
    if padding == 0: padding = 1
    
    fig.update_yaxes(range=[min_val - padding, max_val + padding])

    # CHANGED TITLE TO INCLUDE COUNTRY AND PORT
    fig.update_layout(
        title=f"Impact Analysis: {feature} in {port}, {country}",
        height=550, 
        template="plotly_white", 
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def save_static_plot(results, feature, event_date, country, port, save_path):
    """Generates a static matplotlib chart exclusively for the PDF export."""
    train = results['train']
    test = results['test']
    forecast = results['forecast']
    
    # Bridge the visual gap
    anchor_date = [train.index[-1]]
    anchor_val = [train.iloc[-1]]
    
    test_x = anchor_date + test.index.tolist()
    test_y = anchor_val + test.values.tolist()
    
    forecast_x = anchor_date + forecast.index.tolist()
    forecast_y = anchor_val + forecast.values.tolist()
    
    conf_lower = anchor_val + results['conf_lower'].tolist()
    conf_upper = anchor_val + results['conf_upper'].tolist()

    # Create the Plot
    plt.figure(figsize=(10, 5))
    sns.set_style("whitegrid")
    
    # 1. Historical
    plt.plot(train.index, train.values, label='Historical Data', color='#1f77b4', linewidth=2)
    
    # 2. Actual Post-Event
    plt.plot(test_x, test_y, label='Actual Data', color='#000000', linewidth=2.5)
    
    # 3. Forecast
    plt.plot(forecast_x, forecast_y, label='Expected (SARIMA)', color='#d62728', linewidth=2, linestyle='--')
    
    # 4. Confidence Intervals
    plt.fill_between(forecast_x, conf_lower, conf_upper, color='#d62728', alpha=0.15, label='95% CI')
    
    # 5. Event Line
    event_dt = pd.to_datetime(event_date)
    plt.axvline(x=event_dt, color='gray', linestyle='--', linewidth=2)
    plt.text(event_dt, plt.ylim()[1], ' Event', color='black', ha='left', va='top')
    
    # Y-axis bounds
    max_val = max(train.max(), test.max(), forecast.max())
    min_val = min(train.min(), test.min(), forecast.min())
    padding = (max_val - min_val) * 0.15 if (max_val - min_val) != 0 else max_val * 0.15
    if padding == 0: padding = 1
    plt.ylim(min_val - padding, max_val + padding)
    
    # Formatting
    plt.title(f"Impact Analysis: {feature} in {port}, {country}", fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.tight_layout()
    
    # Save to the temporary path provided by Streamlit
    plt.savefig(save_path, format='png', dpi=150, bbox_inches='tight')
    plt.close()
