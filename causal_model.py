import pandas as pd
from causalimpact import CausalImpact
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def run_causal_analysis(df, feature, event_date, pre_months, post_days):
    """Runs the Bayesian Structural Time Series model."""
    data = df[[feature]].copy()
    event_dt = pd.to_datetime(event_date)
    
    # Calculate time windows
    start_pre = event_dt - pd.DateOffset(months=pre_months)
    end_pre = event_dt - pd.Timedelta(days=1)
    end_post = event_dt + pd.Timedelta(days=post_days)
    
    # Ensure dates are within dataset bounds
    if start_pre < data.index.min(): start_pre = data.index.min()
    if end_post > data.index.max(): end_post = data.index.max()
    
    pre_period = [start_pre.strftime('%Y-%m-%d'), end_pre.strftime('%Y-%m-%d')]
    post_period = [event_dt.strftime('%Y-%m-%d'), end_post.strftime('%Y-%m-%d')]
    
    try:
        ci = CausalImpact(data, pre_period, post_period)
        return ci, pre_period, post_period
    except Exception as e:
        return None, str(e), None

def plot_impact_dashboard(ci, feature, event_date):
    """Generates an interactive Plotly dashboard for the client."""
    inferences = ci.inferences
    
    # Reconstruct the actual data mathematically: Actual = Expected + Difference
    actual_y = inferences['preds'] + inferences['point_effects']
    
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, 
        subplot_titles=(f"Actual vs. Expected {feature}", f"Net Daily Impact (Lost/Gained {feature})"),
        vertical_spacing=0.1,
        row_heights=[0.6, 0.4]
    )

    # Top Chart: Actual vs Expected
    fig.add_trace(go.Scatter(
        x=inferences.index, y=actual_y, 
        name='Actual Data', line=dict(color='#1f77b4', width=2)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=inferences.index, y=inferences['preds'], 
        name='Expected (Counterfactual)', line=dict(color='#d62728', width=2, dash='dash')
    ), row=1, col=1)
    
    # Confidence Interval Shading
    fig.add_trace(go.Scatter(
        x=inferences.index.tolist() + inferences.index[::-1].tolist(),
        y=inferences['preds_upper'].tolist() + inferences['preds_lower'][::-1].tolist(),
        fill='toself', fillcolor='rgba(214, 39, 40, 0.15)', line=dict(color='rgba(255,255,255,0)'),
        name='Expected Range', showlegend=False
    ), row=1, col=1)

    # Bottom Chart: Pointwise Effect (Net difference)
    colors = ['#d62728' if val < 0 else '#2ca02c' for val in inferences['point_effects']]
    fig.add_trace(go.Bar(
        x=inferences.index, y=inferences['point_effects'], 
        marker_color=colors, name='Daily Impact'
    ), row=2, col=1)

    # Convert event_date to a string so Plotly can parse it for the x-axis
    event_date_str = pd.to_datetime(event_date).strftime('%Y-%m-%d')

    # Add Vertical Event Lines to both subplots using the separated text fix
    for row in [1, 2]:
        fig.add_vline(x=event_date_str, line_width=2, line_dash="dash", line_color="black", row=row, col=1)
        
        # Add the text annotation manually to the top chart
        if row == 1:
            fig.add_annotation(
                x=event_date_str, 
                y=1.02, # Place it just slightly above the chart
                yref="paper", 
                text="Event Occurred", 
                showarrow=False, 
                xanchor="left",
                font=dict(color="black", size=12),
                row=row, col=1
            )
    
    fig.update_layout(
        height=700, 
        template="plotly_white", 
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig
