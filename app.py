import streamlit as st
import pandas as pd
from datetime import date
import traceback

from data_loader import fetch_portwatch_countries, preprocess_portwatch_data, get_available_ports
from sarima_analysis import run_sarima_impact_analysis, plot_sarima_dashboard

st.set_page_config(page_title="Maritime Event Impact", layout="wide")

st.title("🌊 Maritime Event Impact Analyzer")
st.markdown("Measure the exact impact of geopolitical, weather, or economic events using Auto-SARIMA forecasting.")

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
        
        resolution = st.selectbox("Data Resolution:", ["Daily", "Weekly", "Monthly"])
        
        default_date = processed_df.index.max() - pd.Timedelta(days=30) if not processed_df.empty else date.today()
        event_date = st.date_input("Date of Disruption/Event:", value=default_date)
        
        st.write("Time Windows:")
        pre_months = st.slider("Historical Data to Learn From (Months):", 1, 36, 12, 
                               help="How much history should the SARIMA model use to learn 'normal' behavior?")
        post_days = st.slider("Time to Measure Impact (Days):", 7, 180, 30,
                              help="How many days after the event do you want to calculate the impact for?")

        run_btn = st.button("📊 Generate Impact Report", type="primary", use_container_width=True)

# --- MAIN PAGE LOGIC ---
if raw_df.empty:
    st.info("👈 Please select a country from the sidebar to begin.")
else:
    if run_btn:
        with st.spinner(f"Training Auto-SARIMA on {resolution} data..."):
            try:
                # Run the math
                results = run_sarima_impact_analysis(
                    processed_df, feature, event_date, pre_months, post_days, resolution
                )
                
                # Calculate basic impact statistics
                total_expected = results['forecast'].sum()
                total_actual = results['test'].sum()
                absolute_diff = total_actual - total_expected
                pct_diff = (absolute_diff / total_expected) * 100 if total_expected != 0 else 0
                
                # 1. Headline Takeaway
                st.divider()
                col1, col2, col3 = st.columns(3)
                col1.metric("Actual Post-Event", f"{total_actual:,.0f}")
                col2.metric("Expected (SARIMA)", f"{total_expected:,.0f}")
                col3.metric("Net Impact", f"{absolute_diff:+,.0f}", f"{pct_diff:+.1f}%")
                
                if absolute_diff < 0:
                    st.error("### 📉 Negative Impact Detected")
                else:
                    st.success("### 📈 Positive / Neutral Impact Detected")
                
                # 2. Visual Dashboard
                st.plotly_chart(plot_sarima_dashboard(results, feature, event_date), use_container_width=True)
                
                # 3. Plain English Report
                st.subheader("📋 Executive Summary")
                st.markdown(f"""
                Based on historical trends, the Auto-SARIMA model expected **{total_expected:,.0f}** {feature} 
                during the {post_days}-day period following the event. 
                
                The actual recorded number was **{total_actual:,.0f}**. 
                This represents a net difference of **{absolute_diff:+,.0f}** ({pct_diff:+.1f}%).
                """)
                
                # Show the automatically inferred seasonality
                if results['inferred_m'] > 1:
                    st.info(f"⚙️ **Model Insight:** The AI automatically detected a recurring pattern every **{results['inferred_m']} periods** and adjusted its forecast accordingly.")
                else:
                    st.info("⚙️ **Model Insight:** No strong recurring seasonal patterns were detected in the historical data.")
                
                # 4. Raw Math Details
                with st.expander("View Auto-SARIMA Model Architecture Details"):
                    st.text(results['model_summary'])
                        
            except ValueError as ve:
                st.warning(f"Data Warning: {ve}")
            except Exception as e:
                st.error("An error occurred during execution.")
                st.code(traceback.format_exc())
