""" 

An application for Time Series Decomposition and basic Forecasting

"""

from datetime import datetime
from turtle import title
import numpy as np
import pandas as pd
from pyparsing import col 
import seaborn as sns 
import streamlit as st 
import matplotlib.pyplot as plt 
from st_aggrid import AgGrid
from statsmodels.tsa.seasonal import seasonal_decompose
import hvplot.pandas as hv
from streamlit_echarts import st_pyecharts
from pyecharts.charts import Bar, Line, Grid
import altair as alt

plt.style.use('ggplot')


# PAGE CONFIGURATIONS
st.set_page_config(layout='wide')


# # HEADER SECTION
with st.container():
    st.header('Time Series Analysis and Forecasting Tool')
    st.write('Perform time series analysis including decomposition, differencing and modelling.')



ts_file = st.sidebar.file_uploader('Upload a time series data file (csv, txt):')


@st.cache(allow_output_mutation=True)
def get_data(ts_file):
    if ts_file is not None:
        df = pd.read_csv(ts_file)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    
    return ''


def decompose_ts(ts_data, metric, model, period):
    ts_series = ts_data[['Date', metric]].copy()
    ts_series.set_index('Date', inplace=True)
    return seasonal_decompose(ts_series, model=model, period=period)



ts_data = get_data(ts_file)


if ts_file:
    
    viz, decomposition, data = st.tabs(['Visualization', 'Decomposition', 'Data'])

    if isinstance(ts_data, pd.DataFrame):

        with viz:
            cols = [col for col in ts_data.columns if col != 'Date']
            plot_df = ts_data.copy()
            plot_df.set_index('Date', inplace=True)
            
            y_values = st.multiselect('Choose Metrics', cols, cols[0])

            with st.container():
                st.line_chart(plot_df[y_values], height=1000, width=500)



        with decomposition:
            metric_col, model_col, period_col = st.columns(3)
            with metric_col:
                metric = st.selectbox('Choose Metric', [col for col in ts_data.columns if col != 'Date'])
            with model_col:
                model = st.selectbox('Choose Model:', ['additive', 'multiplicative'])
            with period_col:
                period = st.number_input('Select Seasonal Period', min_value=1, max_value=365, value=7)
            
            # Implement decomposition
            decomposition = decompose_ts(ts_data, metric, model, period)
            decomposed_df = pd.concat([ decomposition.observed,
                       decomposition.trend,
                       decomposition.seasonal,
                       decomposition.resid ], axis=1, join='outer')
            decomposed_df.columns = ['observed', 'trend', 'seasonal', 'resid']


            observations = alt.Chart(decomposed_df.reset_index()).mark_line().encode(x='Date:T', y='observed:Q')
            trend = alt.Chart(decomposed_df.reset_index()).mark_line().encode(x='Date:T', y='trend')
            seasonal = alt.Chart(decomposed_df.reset_index()).mark_line().encode(x='Date:T', y='seasonal')
            resid = alt.Chart(decomposed_df.reset_index()).mark_line().encode(x='Date:T', y='resid')
            
           
            obs_trend, season_resid  = alt.hconcat(observations, trend), alt.hconcat(seasonal, resid)
            combined = alt.vconcat(obs_trend, season_resid) 
            
            st.altair_chart(combined, use_container_width=True)


        # DISPLAY DATA
        with data:
            AgGrid(ts_data)
