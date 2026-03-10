import pandas as pd
import pmdarima as pm
import plotly.graph_objects as go

def run_sarima_impact_analysis(df, feature, event_date, pre_months, post_days, resolution):
    """Resamples data, trains Auto-SARIMA on pre-event data, and forecasts post-event."""
    data = df[[feature]].copy()
    event_dt = pd.to_datetime(event_date)
    
    # 1. Resample based on user resolution
    if resolution == 'Weekly':
        data = data.resample('W').sum()
        seasonality_m = 52 # 52 weeks in a year
    elif resolution == 'Monthly':
        data = data.resample('ME').sum()
        seasonality_m = 12 # 12 months in a year
    else: # Daily
        data = data.resample('D').sum()
        seasonality_m = 7  # 7 days in a week
        
    # 2. Calculate time windows
    start_pre = event_dt - pd.DateOffset(months=pre_months)
    end_post = event_dt + pd.Timedelta(days=post_days)
    
    # Filter full dataset to the requested window
    data = data[(data.index >= start_pre) & (data.index <= end_post)]
    
    if data.empty:
        raise ValueError("Not enough data in the selected date range.")

    # 3. Split into Train (Pre-Event) and Test (Post-Event)
    train = data[data.index < event_dt][feature]
    test = data[data.index >= event_dt][feature]
    
    if len(train) < 10:
        raise ValueError(f"Not enough historical data ({len(train)} points) to train SARIMA. Increase the 'Months of history' slider.")
    if len(test) == 0:
        raise ValueError("No post-event data available for the selected dates.")
    
    # Disable seasonality if we don't have enough data points (need at least 2 full cycles)
    seasonal = True
    if len(train) < 2 * seasonality_m:
        seasonal = False

    # 4. Auto-SARIMA Training (Finds best p, d, q parameters automatically)
    model = pm.auto_arima(
        train, 
        seasonal=seasonal, 
        m=seasonality_m if seasonal else 1,
        suppress_warnings=True, 
        error_action='ignore',
        trace=False
    )
    
    # 5. Forecast the counterfactual
    forecast, conf_int = model.predict(n_periods=len(test), return_conf_int=True, alpha=0.05)
    
    # Package results
    results = {
        'train': train,
        'test': test,
        'forecast': pd.Series(forecast, index=test.index),
        'conf_lower': pd.Series(conf_int[:, 0], index=test.index),
        'conf_upper': pd.Series(conf_int[:, 1], index=test.index),
        'model_summary': str(model.summary())
    }
    return results

def plot_sarima_dashboard(results, feature, event_date):
    """Generates a clean, single-panel Plotly chart comparing Actuals to SARIMA Forecast."""
    
    train = results['train']
    test = results['test']
    forecast = results['forecast']
    
    fig = go.Figure()

    # 1. Plot Actual Historical Data (Pre-event)
    fig.add_trace(go.Scatter(
        x=train.index, y=train.values, 
        name='Historical Data (Pre-Event)', line=dict(color='#1f77b4', width=2)
    ))
    
    # 2. Plot Actual Post-Event Data
    fig.add_trace(go.Scatter(
        x=test.index, y=test.values, 
        name='Actual Data (Post-Event)', line=dict(color='#ff7f0e', width=2)
    ))

    # 3. Plot SARIMA Forecast (Counterfactual)
    fig.add_trace(go.Scatter(
        x=forecast.index, y=forecast.values, 
        name='Expected (SARIMA Forecast)', line=dict(color='#d62728', width=2, dash='dash')
    ))

    # 4. Confidence Interval Shading
    fig.add_trace(go.Scatter(
        x=forecast.index.tolist() + forecast.index[::-1].tolist(),
        y=results['conf_upper'].tolist() + results['conf_lower'][::-1].tolist(),
        fill='toself', fillcolor='rgba(214, 39, 40, 0.15)', line=dict(color='rgba(255,255,255,0)'),
        name='Expected Range (95% CI)', showlegend=False
    ))

    # 5. Add Vertical Event Line (Using the string conversion to prevent Plotly bugs)
    event_date_str = pd.to_datetime(event_date).strftime('%Y-%m-%d')
    fig.add_vline(x=event_date_str, line_width=2, line_dash="dash", line_color="black")
    
    fig.add_annotation(
        x=event_date_str, y=1.02, yref="paper", 
        text="Event Occurred", showarrow=False, xanchor="left",
        font=dict(color="black", size=12)
    )

    # 6. Lock the Y-Axis to the actual data lines (prevents massive confidence intervals from zooming out too far)
    max_val = max(train.max(), test.max(), forecast.max())
    min_val = min(train.min(), test.min(), forecast.min())
    padding = (max_val - min_val) * 0.15 if (max_val - min_val) != 0 else max_val * 0.15
    if padding == 0: padding = 1
    
    fig.update_yaxes(range=[min_val - padding, max_val + padding])

    fig.update_layout(
        title=f"Impact Analysis: Actual vs. Expected {feature}",
        height=550, 
        template="plotly_white", 
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig
