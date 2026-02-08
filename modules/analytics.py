# -*- coding: utf-8 -*-
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

class AnalyticsEngine:
    """
    Generates interactive charts for the STEMI Dashboard.
    Adapted for 8-feature STEMI model (Removed Gender plots).
    """
    
    def __init__(self, db_manager):
        self.db = db_manager

    def get_data(self):
        return self.db.fetch_all_records()

    def plot_risk_distribution(self):
        """Pie chart: High Risk vs Low Risk"""
        df = self.get_data()
        if df.empty: return None
        
        fig = px.pie(
            df, 
            names='risk_label', 
            title='Cohort Risk Stratification',
            color='risk_label',
            color_discrete_map={'High Risk':'#dc3545', 'Low Risk':'#28a745'},
            hole=0.4
        )
        fig.update_layout(height=350)
        return fig

    def plot_age_distribution(self):
        """Histogram: Age Distribution by Risk Group (Replaces Gender Stats)"""
        df = self.get_data()
        if df.empty or 'Age' not in df.columns: return None
        
        fig = px.histogram(
            df, 
            x="Age", 
            color="risk_label",
            title="Age Distribution by Risk Group",
            nbins=10,
            barmode='overlay',
            color_discrete_map={'High Risk':'#dc3545', 'Low Risk':'#28a745'}
        )
        fig.update_layout(height=350)
        return fig

    def plot_stent_stats(self):
        """Bar chart: Stent Type Usage"""
        df = self.get_data()
        if df.empty or 'Stent for IRA' not in df.columns: return None
        
        # Ensure column matches what is saved in DB (Stent for IRA)
        fig = px.histogram(
            df,
            x="Stent for IRA",
            color="risk_label",
            title="Risk by Stent Strategy",
            barmode='group',
            color_discrete_map={'High Risk':'#dc3545', 'Low Risk':'#28a745'}
        )
        fig.update_layout(height=350)
        return fig

    def plot_temporal_trend(self):
        """Line chart: Average Risk over time"""
        df = self.get_data()
        if df.empty: return None
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        
        daily_avg = df.groupby('date')['risk_prob'].mean().reset_index()
        
        fig = px.line(
            daily_avg, 
            x='date', 
            y='risk_prob',
            title='Daily Average Mortality Risk Trend',
            markers=True
        )
        fig.update_layout(height=350, yaxis_tickformat='.0%')
        return fig
