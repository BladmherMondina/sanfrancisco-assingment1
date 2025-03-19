import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def perform_spatial_clustering(df, eps=0.01, min_samples=5):
    """Perform DBSCAN clustering on spatial data"""
    try:
        # Prepare the data
        coords = df[['latitude', 'longitude']].values
        
        # Scale the coordinates
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(coords)
        
        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords_scaled)
        
        # Add cluster labels to dataframe
        df['cluster'] = clustering.labels_
        
        return df
    except Exception as e:
        raise Exception(f"Error in spatial clustering: {str(e)}")

def detect_anomalies(df, contamination=0.1):
    """Detect anomalies using Isolation Forest"""
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found for anomaly detection")
            
        clf = IsolationForest(contamination=contamination, random_state=42)
        df['is_anomaly'] = clf.fit_predict(df[numeric_cols])
        return df
    except Exception as e:
        raise Exception(f"Error in anomaly detection: {str(e)}")

def mine_association_rules(df, min_support=0.01, min_confidence=0.5):
    """Mine association rules from categorical data"""
    try:
        # Convert categorical columns to one-hot encoding
        categorical_cols = df.select_dtypes(include=['object']).columns
        df_encoded = pd.get_dummies(df[categorical_cols])
        
        # Generate frequent itemsets
        frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
        
        # Generate association rules
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        
        return rules
    except Exception as e:
        raise Exception(f"Error in association rule mining: {str(e)}")
