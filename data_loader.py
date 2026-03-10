import pandas as pd
import requests
import streamlit as st

@st.cache_data(ttl=3600)
def fetch_portwatch_countries():
    """Fetches available datasets from the PortWatch API."""
    try:
        url = "https://data.humdata.org/api/3/action/package_search?q=organization:portwatch&rows=300"
        response = requests.get(url).json()
        datasets = response.get('result', {}).get('results', [])
        
        country_links = {}
        for d in datasets:
            title = d.get('title', '')
            if ": Daily Port Activity" in title:
                country = title.split(":")[0].strip()
                for res in d.get('resources', []):
                    if res.get('format', '').upper() == 'CSV':
                        country_links[country] = res.get('url')
                        break
        return dict(sorted(country_links.items()))
    except:
        return {"Yemen": "https://data.humdata.org/dataset/yemen-daily-port-activity-data-and-shipment-estimates/resource/5b130c44-8c57-485c-b1bf-6ab07aa12ce9/download/yemen-daily-port-activity-data-and-shipment-estimates.csv"}

def preprocess_portwatch_data(df: pd.DataFrame, target_port: str = "All Ports (Sum)") -> pd.DataFrame:
    """Preprocesses data to ensure continuous daily frequency."""
    if df.empty:
        return df
        
    df_clean = df.copy()
    
    # 1. Handle date column
    if 'date' in df_clean.columns:
        df_clean['date'] = pd.to_datetime(df_clean['date']).dt.tz_localize(None)
        df_clean = df_clean.set_index('date')
    elif not pd.api.types.is_datetime64_any_dtype(df_clean.index):
        try:
            df_clean.index = pd.to_datetime(df_clean.index).tz_localize(None)
        except Exception:
            raise ValueError("Dataframe must contain a recognizable date column or index.")
            
    # 2. Filter by port
    if target_port != "All Ports (Sum)" and 'portname' in df_clean.columns:
        df_clean = df_clean[df_clean['portname'] == target_port]
        
    # 3. Aggregate numeric features
    exclude_cols = ['year', 'month', 'day', 'unnamed: 0']
    numeric_cols = [c for c in df_clean.select_dtypes(include='number').columns if c.lower() not in exclude_cols]
    
    df_agg = df_clean.groupby(df_clean.index)[numeric_cols].sum()
    
    # 4. Enforce continuous daily frequency (Fill empty days with 0)
    if not df_agg.empty:
        df_agg.index = df_agg.index.normalize()
        df_agg = df_agg.groupby(df_agg.index).sum()
        
        full_idx = pd.date_range(start=df_agg.index.min(), end=df_agg.index.max(), freq='D')
        df_agg = df_agg.reindex(full_idx, fill_value=0)
        
    return df_agg

def get_available_ports(df: pd.DataFrame) -> list:
    """Extracts unique ports from the dataset."""
    if 'portname' in df.columns:
        return sorted(df['portname'].dropna().astype(str).unique().tolist())
    return []
