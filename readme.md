Overview
Uses streamlit for the visualization aspect of this assignment
specifically on data mining and analysis on San Francisco datasets 
I used spatial clustering and wanted to implement anomaly detection, 
and association rule but I ran out of time and it got frustrating. 
I mostly focused on the police incident reports for the spatial clustering.

Data Sources
Crime data (Police Department Incident Reports)

Features
Spatial Clustering
* uses DBSCAN algorithm to identify geographic cluster
* interactive map visualization with cluster highlighting
* density heatmap generation (not working just wanted to inlcude it if i got it working)
* adjustable clustering parameters (radius and minimum samples)

Installation
Install the required packages:
pip install streamlit pandas numpy scikit-learn mlxtend plotly folium streamlit-folium sodapy

Run the application:
python3 app.py
streamlit run app.py

What my files do:
data_loader.py
* uses soapy to connect to the san francisco database with open data api
* implements data preprocessing and cleaning
* handles location data extraction and formatting
* (would provide specific loaders for each dataset type)

analysis.py
I am just gonna cover only the spatial clustering for this since the others didn't work
* DBSCAN implementation for geographic clustering
* Standardized coordinate scaling
* Cluster label assignment

visualization.py
* Interactive maps using Folium
the ones with () are excluded
* (Heatmaps with Plotly)
* (Cluster visualization)
* (Association rule scatter plots)

app.py
this will be my demo 
* offers options for 3 different datasets and different data mining analysis
* currently the other 2 data mining analysis options do not work