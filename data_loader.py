import pandas as pd
import numpy as np
from sodapy import Socrata
4
def load_sf_dataset(dataset_id, limit=10000):
    """Load dataset from SF Open Data Portal"""
    try:
        client = Socrata("data.sfgov.org", None)
        results = client.get(dataset_id, limit=limit)
        df = pd.DataFrame.from_records(results)
        return df
    except Exception as e:
        raise Exception(f"Error loading dataset {dataset_id}: {str(e)}")

def preprocess_location_data(df):
    """Preprocess location data for analysis"""
    if 'location' in df.columns:
        # Extract latitude and longitude from location field
        df['latitude'] = df['location'].apply(lambda x: float(x['latitude']) if x and 'latitude' in x else np.nan)
        df['longitude'] = df['location'].apply(lambda x: float(x['longitude']) if x and 'longitude' in x else np.nan)
    
    # Handle missing values
    df = df.dropna(subset=['latitude', 'longitude'])
    return df

def load_crime_data():
    """Load and preprocess SF crime data"""
    crime_df = load_sf_dataset("wg3w-h783")  # SF Police Department Incident Reports
    crime_df = preprocess_location_data(crime_df)
    return crime_df

def load_business_data():
    """Load and preprocess SF business data"""
    business_df = load_sf_dataset("g8m3-pdis")  # Registered Business Locations
    business_df = preprocess_location_data(business_df)
    return business_df

def load_311_data():
    """Load and preprocess SF 311 cases data"""
    cases_df = load_sf_dataset("vw6y-z8j6")  # 311 Cases
    cases_df = preprocess_location_data(cases_df)
    return cases_df
