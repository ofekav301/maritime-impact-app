import streamlit as st
import pandas as pd
from datetime import date
import traceback

from data_loader import fetch_portwatch_countries, preprocess_portwatch_data, get_available_ports
from causal_model import run_causal_analysis, plot_impact_dashboard

st.set_page_config(page_title="Maritime Event Impact", layout="wide")

st.title("🌊 Maritime Event Impact Analyzer")
st.markdown("Measure the exact impact of geopolitical, weather, or economic events on port activity.")

# --- SIDEBAR UI ---
with st.sidebar:
    st.header("1. Data Selection")
    country_dict = fetch_portwatch_countries()
    dataset_options = ["-- Select a Country --"] + list(country_dict.keys())
    selection = st.selectbox("Country:", dataset_options)
    
    raw_df = pd.DataFrame()
    if selection != "-- Select a Country --":
        url_to_fetch = country_dict.get(selection)
        if url_to_fetch:
            try:
                raw_df = pd.read_csv(url_to_fetch, parse_dates=True)
                st.success(f"Loaded {len(raw_df)} rows")
            except Exception as e:
                st.error(f"Error loading CSV: {e}")

    if not raw_df.empty:
        available_ports = get_available_ports(raw_df)
        port_options = ["All Ports (Sum)"] + available_ports if available_ports else ["All Ports (Sum)"]
        target_port = st.selectbox("Target Port:", port_options)
        
        processed_df = preprocess_portwatch_data(raw_df, target_port=target_port)
        numeric_cols = processed_df.select_dtypes(include='number').columns.tolist()
        feature = st.selectbox("Metric to Analyze:", numeric_cols)
        
        st.header("2. Event Configuration")
        # Default event date roughly 30 days before the end of the dataset
        default_date = processed_df.index.max() - pd.Timedelta(days=30) if not processed_df.empty else date.today()
        event_date = st.date_input("Date of Disruption/Event:", value=default_date)
        
        st.write("Time Windows:")
        pre_months = st.slider("Historical Data to Learn From (Months):", 1, 36, 12, 
                               help="How much history should the AI use to learn 'normal' behavior?")
        post_days = st.slider("Time to Measure Impact (Days):", 7, 180, 30,
                              help="How many days after the event do you want to calculate the impact for?")

        run_btn = st.button("📊 Generate Impact Report", type="primary", use_container_width=True)

# --- MAIN PAGE LOGIC ---
if raw_df.empty:
    st.info("👈 Please select a country from the sidebar to begin.")
else:
    if run_btn:
        with st.spinner(f"Calculating true causal impact on {feature}..."):
            try:
                ci, pre_period, post_period = run_causal_analysis(
                    processed_df, feature, event_date, pre_months, post_days
                )
                
                if ci is None:
                    st.error(f"Analysis failed. Ensure you have enough data before and after the event date. {pre_period}")
                else:
                    # 1. Headline Takeaway
                    st.divider()
                    p_value = ci.p_value
                    is_significant = p_value < 0.05
                    
                    if is_significant:
                        st.error(f"### 🚨 Significant Impact Detected")
                        st.markdown(f"The event on **{event_date}** had a statistically significant impact on **{feature}** (p-value: {p_value:.3f}).")
                    else:
                        st.success(f"### ✅ No Significant Impact")
                        st.markdown(f"Any fluctuations in **{feature}** after **{event_date}** were within normal, expected bounds (p-value: {p_value:.3f}).")
                    
                    # 2. Visual Dashboard
                    st.plotly_chart(plot_impact_dashboard(ci, feature, event_date), use_container_width=True)
                    
                    # 3. Plain English Report
                    st.subheader("📋 Executive Summary")
                    st.info(ci.summary(output='report'))
                    
                    # 4. Raw Numbers
                    with st.expander("View Mathematical Summary"):
                        st.text(ci.summary())
                        
            except Exception as e:
                st.error("An error occurred during execution.")
                st.code(traceback.format_exc())
