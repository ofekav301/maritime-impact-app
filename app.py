import streamlit as st
import pandas as pd
from datetime import date
import tempfile
import os
import traceback

from data_loader import fetch_portwatch_countries, preprocess_portwatch_data, get_available_ports
from prophet_analysis import run_prophet_impact_analysis, plot_impact_dashboard, save_static_plot
from reporting import create_impact_pdf_report

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
                # 1. Run the math
                results = run_prophet_impact_analysis(
                    processed_df, feature, event_date, pre_months, post_days, resolution
                )
                
                # 2. Calculate impact stats & Confidence Intervals
                total_expected = results['forecast'].sum()
                total_actual = results['test'].sum()
                absolute_diff = total_actual - total_expected
                pct_diff = (absolute_diff / total_expected) * 100 if total_expected != 0 else 0
                
                # Sum the confidence intervals to get the "Normal Bounds"
                total_lower_bound = results['conf_lower'].sum()
                total_upper_bound = results['conf_upper'].sum()
                
                st.divider()

                # 3. Three-way Headline Logic (Positive, Negative, No Effect)
                if total_actual > total_upper_bound:
                    st.markdown(f"<h2 style='color: #2ca02c;'>📈 Positive Impact Detected following {event_date}</h2>", unsafe_allow_html=True)
                    impact_text = "exceeded the expected upper bounds"
                    impact_headline = "Positive Impact"
                elif total_actual < total_lower_bound:
                    st.markdown(f"<h2 style='color: #d62728;'>📉 Negative Impact Detected following {event_date}</h2>", unsafe_allow_html=True)
                    impact_text = "fell below the expected lower bounds"
                    impact_headline = "Negative Impact"
                else:
                    st.markdown(f"<h2 style='color: #7f7f7f;'>➖ No Significant Effect Detected following {event_date}</h2>", unsafe_allow_html=True)
                    impact_text = "remained within normal expected bounds"
                    impact_headline = "No Significant Effect"
                
                # 4. Clarified Metrics Block
                st.caption(f"Measurements represent the **cumulative sum** of {feature} over the {post_days}-day post-event window.")
                col1, col2, col3 = st.columns(3)
                col1.metric("Cumulative Actual (Sum)", f"{total_actual:,.0f} {feature}")
                col2.metric("Cumulative Expected (Sum)", f"{total_expected:,.0f} {feature}")
                col3.metric("Net Impact (Sum)", f"{absolute_diff:+,.0f} {feature}", f"{pct_diff:+.1f}%")
                
                # 5. Visual Dashboard (Interactive Plotly for the Web UI)
                fig = plot_impact_dashboard(results, feature, event_date, selection, target_port)
                st.plotly_chart(fig, use_container_width=True)
                
                # 6. Plain English Report
                st.subheader("📋 Executive Summary")
                exec_summary = (
                    f"Based on historical trends, the Auto-SARIMA model expected a cumulative total of "
                    f"**{total_expected:,.0f}** {feature} during the {post_days}-day period following the event. "
                    f"The actual recorded number was **{total_actual:,.0f}**, which {impact_text}. "
                    f"This represents a net cumulative difference of **{absolute_diff:+,.0f}** ({pct_diff:+.1f}%)."
                )
                st.markdown(exec_summary)
                
                if results['inferred_m'] > 1:
                    st.info(f"⚙️ **Model Insight:** The AI automatically detected a recurring pattern every **{results['inferred_m']} periods** and adjusted its forecast accordingly.")
                else:
                    st.info("⚙️ **Model Insight:** No strong recurring seasonal patterns were detected in the historical data.")

                # 7. PDF Report Generation
                st.subheader("📥 Export")
                with st.spinner("Generating PDF Report..."):
                    
                    # Temporarily save the MATPLOTLIB static chart as a PNG
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                        img_path = tmpfile.name
                        # Using our fallback matplotlib function to avoid Kaleido/Chromium issues!
                        save_static_plot(results, feature, event_date, selection, target_port, img_path)

                    pdf_bytes = create_impact_pdf_report(
                        country=selection, port=target_port, feature=feature, 
                        event_date=event_date, resolution=resolution, 
                        total_actual=total_actual, total_expected=total_expected, 
                        absolute_diff=absolute_diff, pct_diff=pct_diff, 
                        impact_text=impact_headline, exec_summary=exec_summary, 
                        img_path=img_path
                    )
                    
                    # Clean up the temporary image
                    if os.path.exists(img_path):
                        os.remove(img_path)

                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"ofek_impact_{selection}_{event_date}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

                with st.expander("View Auto-SARIMA Model Architecture Details"):
                    st.text(results['model_summary'])
                        
            except ValueError as ve:
                st.warning(f"Data Warning: {ve}")
            except Exception as e:
                st.error("An error occurred during execution.")
                st.code(traceback.format_exc())
