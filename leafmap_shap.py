import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pylab as pl

import shap
import dill

import geopandas

import leafmap
import leafmap.kepler as leafmapp

#st.set_page_config(layout="wide")

st.title('Leafmap')

@st.cache(suppress_st_warning = True, allow_output_mutation=True)
def load_imp():

    with open('shap_values', 'rb') as f:
          shap_values = dill.load(f)

    finaldf = pd.read_csv('finaldf.csv')
    test_X = pd.read_csv('testx.csv')
    test_X = test_X[0:10000]

    s = pd.DataFrame(shap_values)
    s.columns = test_X.columns
    kmeans = KMeans(n_clusters = 3, random_state = 100).fit(s)
    centroids = kmeans.cluster_centers_

    property_data = pd.DataFrame(test_X.index)
    cluster_no = pd.DataFrame(kmeans.labels_)

    df= pd.DataFrame()
    df = pd.concat([property_data,cluster_no], axis =1)
    df.columns = ["property_data", "cluster no"]

    final_df_cluster = pd.concat([finaldf,df], axis = 1)
    #s.reset_index(inplace=True)

    #st.write(s)

    cluster1 = final_df_cluster[final_df_cluster['cluster no']==0]
    cluster2 = final_df_cluster[final_df_cluster['cluster no']==1]
    cluster3 = final_df_cluster[final_df_cluster['cluster no']==2]

    #s = s.drop('index',axis = 1)

    df3_shap = np.array(s.iloc[0])
    predict_cluster = kmeans.predict(df3_shap.reshape(1, -1))

    return [cluster1, cluster2, cluster3, predict_cluster]

def app():

    listofelements = load_imp()
    cluster1 = listofelements[0]
    cluster2 = listofelements[1]
    cluster3 = listofelements[2]
    predict_cluster = listofelements[3]

    if predict_cluster[0] == 0:
            cluster1 = cluster1.reset_index()
            cluster1 = cluster1.drop('index',axis =1)
            m = leafmap.Map()
            m.add_points_from_xy(cluster1[0:10], x='longitude', y='latitude', popup=['ADDRESS','BLOCK','LOT',
                                                                 'RESIDENTIAL UNITS', 'COMMERCIAL UNITS','LAND SQUARE FEET',
                                                                 'GROSS SQUARE FEET', 'PROPERTY AGE','SALE PRICE'],
                                 layer_name="Property",icon_names=['map'],color_column='cluster no')
    elif predict_cluster[0] == 1:
            cluster2 = cluster2.reset_index()
            cluster2 = cluster2.drop('index',axis =1)
            m = leafmap.Map()
            m.add_points_from_xy(cluster2[0:10], x='longitude', y='latitude', popup=['ADDRESS','BLOCK','LOT',
                                                                 'RESIDENTIAL UNITS', 'COMMERCIAL UNITS','LAND SQUARE FEET',
                                                                 'GROSS SQUARE FEET', 'PROPERTY AGE','SALE PRICE'],
                                 layer_name="Property",icon_names=['map'],color_column='cluster no')
    else:
            cluster3 = cluster3.reset_index()
            cluster3 = cluster3.drop('index',axis =1)
            m = leafmap.Map()
            m.add_points_from_xy(cluster3[0:10], x='longitude', y='latitude', popup=['ADDRESS','BLOCK','LOT',
                                                                 'RESIDENTIAL UNITS', 'COMMERCIAL UNITS','LAND SQUARE FEET',
                                                                 'GROSS SQUARE FEET', 'PROPERTY AGE','SALE PRICE'],
                                 layer_name="Property",icon_names=['map'])

    with st.echo():

      m.to_streamlit()

    #mk = leafmapp.Map(center=[40,-70], zoom=12)
    #in_csv = 'sample_data/finaldf.csv'
    #mk.add_csv(in_csv, layer_name="Real Estate")
    #with st.echo():
    #  mk.to_streamlit()
