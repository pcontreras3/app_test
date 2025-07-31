import streamlit as st
import pandas as pd
from utils.functions_st import double_lift_chart, run_cba_manual

st.set_page_config(layout="wide")
st.title("Double Lift Chart & CBA analysis")

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (.csv)", type="csv")

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

if uploaded_file is not None:
    df = load_data(uploaded_file)

    st.sidebar.header("Checkbox to show raw data")
    if st.sidebar.checkbox("Show raw data", value=True):
        st.subheader("Raw data")
        st.write(df)

    # st.subheader("Raw Data")
    # st.write(df)

    # st.subheader("Select Columns to Plot and calculate CBA")
    # col1 = st.selectbox("Select Actual column", df.columns, index=0)  # key="ID")
    # col2 = st.selectbox("Select Expected 1 column", df.columns, index=1) # key="model_a_col")
    # col3 = st.selectbox("Select Expected 2 column", df.columns, index=2) #key="model_b_col")
    
    
    # Sidebar for column selection
    st.sidebar.header("Column selection")
    col1 = st.sidebar.selectbox("Select Actual values", df.columns, index=0)
    col2 = st.sidebar.selectbox("Select Expected 1 values", df.columns, index=1)
    col3 = st.sidebar.selectbox("Select Expected 2 values", df.columns, index=2)


    if all([col1, col2, col3]):
        if pd.api.types.is_numeric_dtype(df[col2]):
            df = df.query(f"{col2} > 0")

            st.subheader("Double Lift Chart")
            double_lift_chart(df, col1, col2, col3)

            st.subheader("CBA")
            run_cba_manual(df, col1, col2, col3, 'nblr', elasticity=2, close_ratio=0.1)
        else:
            st.warning(f"The selected column '{col2}' is not numeric.")




