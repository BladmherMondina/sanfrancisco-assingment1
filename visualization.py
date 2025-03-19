import streamlit as st
import plotly.express as px
import folium
from streamlit_folium import folium_static

def plot_map_clusters(df, zoom=12):
    """Plot clustered data on an interactive map"""
    try:
        # Create base map centered on San Francisco
        m = folium.Map(location=[37.7749, -122.4194], zoom_start=zoom)
        
        # Add points colored by cluster
        for idx, row in df.iterrows():
            color = 'red' if row['cluster'] == -1 else f'#{hash(row["cluster"]) % 0xFFFFFF:06x}'
            folium.CircleMarker(
                [row['latitude'], row['longitude']],
                radius=8,
                color=color,
                fill=True,
                popup=str(row['cluster'])
            ).add_to(m)
            
        return m
    except Exception as e:
        raise Exception(f"Error creating cluster map: {str(e)}")

def plot_heatmap(df):
    """Create a heatmap visualization"""
    try:
        fig = px.density_heatmap(
            df,
            lat='latitude',
            lon='longitude',
            title='Density Heatmap'
        )
        return fig
    except Exception as e:
        raise Exception(f"Error creating heatmap: {str(e)}")

def plot_association_rules(rules):
    """Visualize association rules"""
    try:
        fig = px.scatter(
            rules,
            x='support',
            y='confidence',
            size='lift',
            hover_data=['antecedents', 'consequents'],
            title='Association Rules Analysis'
        )
        return fig
    except Exception as e:
        raise Exception(f"Error creating association rules plot: {str(e)}")
