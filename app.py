import streamlit as st
import pandas as pd
from data_loader import load_crime_data, load_business_data, load_311_data
from streamlit_folium import folium_static
from analysis import perform_spatial_clustering, detect_anomalies, mine_association_rules
from visualization import plot_map_clusters, plot_heatmap, plot_association_rules

def main():
    st.title("San Francisco Data Mining & Analysis")
    
    st.sidebar.header("Analysis Options")
    dataset = st.sidebar.selectbox(
        "Select Dataset",
        ["Crime Data", "Business Locations", "311 Cases"]
    )
    
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Spatial Clustering", "Anomaly Detection", "Association Rules"]
    )
    
    try:
        # Load data based on selection
        with st.spinner("Loading data..."):
            if dataset == "Crime Data":
                df = load_crime_data()
            elif dataset == "Business Locations":
                df = load_business_data()
            else:
                df = load_311_data()
        
        st.write(f"Loaded {len(df)} records")
        
        # Perform selected analysis
        if analysis_type == "Spatial Clustering":
            eps = st.sidebar.slider("Clustering Radius", 0.001, 0.1, 0.01)
            min_samples = st.sidebar.slider("Minimum Samples", 2, 50, 5)
            
            with st.spinner("Performing clustering analysis..."):
                clustered_df = perform_spatial_clustering(df, eps, min_samples)
                st.write(f"Found {len(set(clustered_df['cluster'])) - 1} clusters")
                
                st.subheader("Cluster Map")
                cluster_map = plot_map_clusters(clustered_df)
                folium_static(cluster_map)
                
                st.subheader("Density Heatmap")
                heatmap_fig = plot_heatmap(clustered_df)
                st.plotly_chart(heatmap_fig)
        
        elif analysis_type == "Anomaly Detection":
            contamination = st.sidebar.slider("Anomaly Ratio", 0.01, 0.5, 0.1)
            
            with st.spinner("Detecting anomalies..."):
                anomaly_df = detect_anomalies(df, contamination)
                st.write(f"Found {sum(anomaly_df['is_anomaly'] == -1)} anomalies")
                
                st.subheader("Anomaly Distribution")
                st.write(anomaly_df['is_anomaly'].value_counts())
        
        elif analysis_type == "Association Rules":
            min_support = st.sidebar.slider("Minimum Support", 0.01, 0.5, 0.01)
            min_confidence = st.sidebar.slider("Minimum Confidence", 0.1, 1.0, 0.5)
            
            with st.spinner("Mining association rules..."):
                rules = mine_association_rules(df, min_support, min_confidence)
                st.write(f"Found {len(rules)} association rules")
                
                st.subheader("Association Rules Visualization")
                rules_fig = plot_association_rules(rules)
                st.plotly_chart(rules_fig)
                
                st.subheader("Top Rules")
                st.write(rules.sort_values('lift', ascending=False).head())
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
