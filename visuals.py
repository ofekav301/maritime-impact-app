import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

def plot_impact_dashboard(results, feature, event_date, country, port, model_name):
    """Generates a connected Plotly chart comparing Actuals to the Forecast."""
    train = results['train']
    test = results['test']
    forecast = results['forecast']
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=train.index, y=train.values, name='Historical Data', line=dict(color='#1f77b4', width=2)))
    
    anchor_date = [train.index[-1]]
    anchor_val = [train.iloc[-1]]
    
    test_x = anchor_date + test.index.tolist()
    test_y = anchor_val + test.values.tolist()
    forecast_x = anchor_date + forecast.index.tolist()
    forecast_y = anchor_val + forecast.values.tolist()
    conf_lower = anchor_val + results['conf_lower'].tolist()
    conf_upper = anchor_val + results['conf_upper'].tolist()

    fig.add_trace(go.Scatter(x=test_x, y=test_y, name='Actual Data (Post-Event)', line=dict(color='#000000', width=2.5)))
    fig.add_trace(go.Scatter(x=forecast_x, y=forecast_y, name=f'Expected ({model_name})', line=dict(color='#d62728', width=2, dash='dash')))
    
    fig.add_trace(go.Scatter(
        x=forecast_x + forecast_x[::-1],
        y=conf_upper + conf_lower[::-1],
        fill='toself', fillcolor='rgba(214, 39, 40, 0.15)', line=dict(color='rgba(255,255,255,0)'),
        name='Expected Range (95% CI)', showlegend=False
    ))

    event_date_str = pd.to_datetime(event_date).strftime('%Y-%m-%d')
    fig.add_vline(x=event_date_str, line_width=2, line_dash="dash", line_color="gray")
    fig.add_annotation(x=event_date_str, y=1.02, yref="paper", text="Event Occurred", showarrow=False, xanchor="left", font=dict(color="black", size=12))

    max_val = max(train.max(), test.max(), forecast.max())
    min_val = min(train.min(), test.min(), forecast.min())
    padding = (max_val - min_val) * 0.15 if (max_val - min_val) != 0 else max_val * 0.15
    if padding == 0: padding = 1
    
    fig.update_yaxes(range=[min_val - padding, max_val + padding])
    fig.update_layout(title=f"Impact Analysis: {feature} in {port}, {country}", height=550, template="plotly_white", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    
    return fig

def save_static_plot(results, feature, event_date, country, port, save_path, model_name):
    """Generates a static matplotlib chart exclusively for the PDF export."""
    train = results['train']
    test = results['test']
    forecast = results['forecast']
    
    anchor_date = [train.index[-1]]
    anchor_val = [train.iloc[-1]]
    test_x = anchor_date + test.index.tolist()
    test_y = anchor_val + test.values.tolist()
    forecast_x = anchor_date + forecast.index.tolist()
    forecast_y = anchor_val + forecast.values.tolist()
    conf_lower = anchor_val + results['conf_lower'].tolist()
    conf_upper = anchor_val + results['conf_upper'].tolist()

    plt.figure(figsize=(10, 5))
    sns.set_style("whitegrid")
    
    plt.plot(train.index, train.values, label='Historical Data', color='#1f77b4', linewidth=2)
    plt.plot(test_x, test_y, label='Actual Data', color='#000000', linewidth=2.5)
    plt.plot(forecast_x, forecast_y, label=f'Expected ({model_name})', color='#d62728', linewidth=2, linestyle='--')
    plt.fill_between(forecast_x, conf_lower, conf_upper, color='#d62728', alpha=0.15, label='95% CI')
    
    event_dt = pd.to_datetime(event_date)
    plt.axvline(x=event_dt, color='gray', linestyle='--', linewidth=2)
    plt.text(event_dt, plt.ylim()[1], ' Event', color='black', ha='left', va='top')
    
    max_val = max(train.max(), test.max(), forecast.max())
    min_val = min(train.min(), test.min(), forecast.min())
    padding = (max_val - min_val) * 0.15 if (max_val - min_val) != 0 else max_val * 0.15
    if padding == 0: padding = 1
    plt.ylim(min_val - padding, max_val + padding)
    
    plt.title(f"Impact Analysis: {feature} in {port}, {country}", fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(save_path, format='png', dpi=150, bbox_inches='tight')
    plt.close()
