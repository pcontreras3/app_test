import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize
import streamlit as st

## Double Lift Chart

def double_lift_chart(df, actual_col, model_a_col, model_b_col, factor_adj=1, quantiles=10):
    """
    Plots a Double Lift Chart comparing Model A and Model B predictions against actual values.

    Parameters:
    - df: pandas DataFrame containing the data
    - actual_col: column name for actual loss cost
    - model_a_col: column name for Model A predicted values
    - model_b_col: column name for Model B predicted values
    - factor_adj: factor to adjust baseline, useful when comparing burning cost to TP.
    - quantiles: number of quantile buckets (default is 10 for deciles)
    """
    df = df.copy()
    df['Sort_Ratio'] = df[model_a_col] / df[model_b_col]
    df_sorted = df.sort_values(by='Sort_Ratio')
    df_sorted['Quantile'] = pd.qcut(df_sorted['Sort_Ratio'], quantiles, labels=False, duplicates='drop') + 1

    grouped = df_sorted.groupby('Quantile').agg({
        model_a_col: 'mean',
        model_b_col: 'mean',
        actual_col: 'mean'
    })

    grouped_normalized = grouped.div(grouped[actual_col], axis=0)
    grouped_normalized[actual_col] = grouped_normalized[actual_col] * factor_adj

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=grouped_normalized.index,
        y=grouped_normalized[model_a_col],
        mode='lines+markers',
        name=model_a_col,
        line=dict(color='royalblue', width=3),
        marker=dict(symbol='circle', size=8)
    ))

    fig.add_trace(go.Scatter(
        x=grouped_normalized.index,
        y=grouped_normalized[model_b_col],
        mode='lines+markers',
        name=model_b_col,
        line=dict(color='darkorange', width=3),
        marker=dict(symbol='square', size=8)
    ))

    fig.add_trace(go.Scatter(
        x=grouped_normalized.index,
        y=grouped_normalized[actual_col],
        mode='lines+markers',
        name=actual_col,
        line=dict(color='white', width=3, dash='dash'),
        marker=dict(symbol='diamond', size=8)
    ))

    fig.update_layout(
        title='Double Lift Chart',
        xaxis_title='Quantile (based on Sort Ratio)',
        yaxis_title='Normalized Average Pure Premium',
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(family='Arial', size=14),
        legend=dict(title='Legend', bordercolor='gray', borderwidth=1),
        xaxis=dict(showgrid=True, gridcolor='lightgray',spikecolor='white'),
        yaxis=dict(showgrid=True, gridcolor='lightgray'),
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)


from scipy.optimize import minimize

def run_cba_manual(df, actual, model1_pred, model2_pred, scenario='nblr', elasticity=2.0, close_ratio=0.1):
    curr_pred = df[model1_pred].values
    prop_pred = df[model2_pred].values
    obs_lc = df[actual].values

    def growth_formula(x, curr_pred, prop_pred, elasticity, close_ratio):
        prem1 = prop_pred * x
        ratio = prem1 / curr_pred
        exponent = elasticity / (1 - close_ratio)
        pol1 = 1 / (close_ratio + (1 - close_ratio) * (ratio ** exponent))
        return pol1, prem1

    def calculate_metrics(curr_pred, prop_pred, obs_lc, elasticity, close_ratio, x):
        pol1, prem1 = growth_formula(x, curr_pred, prop_pred, elasticity, close_ratio)
        pol0 = np.ones_like(curr_pred)

        loss0 = np.sum(obs_lc * pol0)
        loss1 = np.sum(obs_lc * pol1)
        nbwp0 = np.sum(curr_pred * pol0)
        nbwp1 = np.sum(prem1 * pol1)
        nbpif0 = np.sum(pol0)
        nbpif1 = np.sum(pol1)

        lr0 = loss0 / nbwp0
        lr1 = loss1 / nbwp1

        metrics = {
            'BaseRate_Offset': x,
            'LR_Impact': lr1 / lr0 - 1,
            'Prem_Impact': nbwp1 / nbwp0 - 1,
            'Wgt_Impact': nbpif1 / nbpif0 - 1,
            'LR_0': lr0,
            'LR_1': lr1,
            'Prem_0': nbwp0,
            'Prem_1': nbwp1,
            'Wgt_0': nbpif0,
            'Wgt_1': nbpif1,
        }
        return metrics

    def loss_nblr(x):
        pol1, prem1 = growth_formula(x, curr_pred, prop_pred, elasticity, close_ratio)
        nbwp0 = np.sum(curr_pred)
        nbwp1 = np.sum(prem1 * pol1)
        return abs(nbwp1 - nbwp0)

    def loss_nbwp(x):
        pol1, prem1 = growth_formula(x, curr_pred, prop_pred, elasticity, close_ratio)
        loss0 = np.sum(obs_lc)
        loss1 = np.sum(obs_lc * pol1)
        nbwp0 = np.sum(curr_pred)
        nbwp1 = np.sum(prem1 * pol1)
        lr0 = loss0 / nbwp0
        lr1 = loss1 / nbwp1
        return abs(lr1 - lr0)

    with st.spinner("Running CBA Optimization..."):
        if scenario == 'nblr':
            res = minimize(loss_nblr, x0=[1.0], method='Nelder-Mead')
        elif scenario == 'nbwp':
            res = minimize(loss_nbwp, x0=[1.0], method='Nelder-Mead')
        else:
            st.error("Escenario debe ser uno de: 'nblr', 'nbwp'")
            return

    x_opt = res.x[0]
    metrics = calculate_metrics(curr_pred, prop_pred, obs_lc, elasticity, close_ratio, x_opt)
    st.success(f"Optimization completed for scenario: {scenario}")
    st.dataframe(pd.DataFrame([metrics], index=[scenario]))
