import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import silhouette_score
import pylab as pl
import shap
import dill
import geopandas
import leafmap.foliumap as leafmap

st.set_page_config(layout="wide")

@st.cache(suppress_st_warning=True)
def display_neighborhoods(df):

    neighborhoods = list(df['neighborhood'].unique())
    bcategory = list(df['building_class_category'].unique())

    neighborhoodsman = []
    neighborhoodsbrook = []
    neighborhoodsbronx = []
    neighborhoodsqueens = []
    neighborhoodsstat = []

    for index,rows in df.iterrows():
      if rows['borough'] == 1:
        neighborhoodsman.append(rows['neighborhood'])
      elif rows['borough'] == 2:
        neighborhoodsbronx.append(rows['neighborhood'])
      elif rows['borough'] == 3:
        neighborhoodsbrook.append(rows['neighborhood'])
      elif rows['borough'] == 4:
        neighborhoodsqueens.append(rows['neighborhood'])
      else:
        neighborhoodsstat.append(rows['neighborhood'])

    neighborhoodsman = set(neighborhoodsman)
    neighborhoodsbronx = set(neighborhoodsbronx)
    neighborhoodsbrook = set(neighborhoodsbrook)
    neighborhoodsqueens = set(neighborhoodsqueens)
    neighborhoodsstat = set(neighborhoodsstat)

    return [neighborhoodsman,neighborhoodsbronx,neighborhoodsbrook,neighborhoodsqueens,neighborhoodsstat,bcategory]

@st.cache(suppress_st_warning = True, allow_output_mutation=True)
def load_imp():

    with open('shap_values', 'rb') as f:
          shap_values = dill.load(f)

    finaldf = pd.read_csv('finaldf.csv')
    dfcopy = pd.read_csv('dfcopy.csv')

    train_X = pd.read_csv('trainx.csv')
    train_Y = pd.read_csv('trainy.csv')
    test_X = pd.read_csv('testx.csv')
    test_Y = pd.read_csv('testy.csv')

    rf_model = RandomForestRegressor()
    rf_model = rf_model.fit(train_X, train_Y.values.ravel())

    shap_exp = shap.TreeExplainer(rf_model)

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
    s.reset_index(inplace=True)

    #st.write(s)
    shap_values_cluster = pd.concat([shap_values,df], axis = 1)

    cluster1 = final_df_cluster[final_df_cluster['cluster no']==0]
    cluster2 = final_df_cluster[final_df_cluster['cluster no']==1]
    cluster3 = final_df_cluster[final_df_cluster['cluster no']==2]

    return [dfcopy, shap_exp, kmeans, cluster1, cluster2, cluster3]

def app():

    listofelements = load_imp()
    dfcopy = listofelements[0]
    shap_exp = listofelements[1]
    kmeans = listofelements[2]
    cluster1 = listofelements[3]
    cluster2 = listofelements[4]
    cluster3 = listofelements[5]

    option = st.selectbox('Borough',
         ('Brooklyn', 'Bronx', 'Manhattan','Queens','Staten Island'))

    listofn = display_neighborhoods(dfcopy)

    if option == 'Brooklyn':
        neighborhoodop = st.selectbox('Neighborhood',(listofn[2]))
    elif option == 'Bronx':
        neighborhoodop = st.selectbox('Neighborhood',(listofn[1]))
    elif option == 'Manhattan':
        neighborhoodop = st.selectbox('Neighborhood',(listofn[0]))
    elif option == 'Queens':
        neighborhoodop = st.selectbox('Neighborhood',(listofn[3]))
    else:
        neighborhoodop = st.selectbox('Neighborhood',(listofn[4]))


    form = st.form(key='my_form')
    resunits = form.number_input(label='Residential Units that your property has',min_value=0)
    comunits = form.number_input(label='Commercial Units that your property has',min_value=0)
    land = form.number_input(label='Land Square Feet Area that your property has',min_value=1000)
    gross = form.number_input(label='Gross Square Feet Area that your property has',min_value=1000)
    age = form.number_input(label='Age of your property',min_value=0)
    tax = form.selectbox('Tax Class',
         ('1','2','4'))
    building = form.selectbox('Building Class Category',(listofn[5]))

    submit_button = form.form_submit_button(label='Submit')

    shapbutton = 1

    if submit_button or shapbutton:

        total = float(resunits)+float(comunits)
        if option == 'Manhattan':
            option = 1
        elif option == 'Bronx':
            option = 2
        elif option == 'Brooklyn':
            option = 3
        elif option == 'Queens':
            option = 4
        else:
            option = 5
        df2 = {'RESIDENTIAL UNITS':[resunits],'COMMERCIAL UNITS':[comunits],'TOTAL UNITS':[total],
        'LAND SQUARE FEET':[land],'GROSS SQUARE FEET':[gross],'PROPERTY AGE':[age],
        'SALE PRICE_B':[option],'SALE PRICE_N':[neighborhoodop],'SALE PRICE_T':[tax],'SALE PRICE_BG_CLASS':[building],
        'SALE PRICE_BOROUGH':[0.0],'SALE PRICE_NEIGHBORHOOD':[0.0],'SALE PRICE_TAX':[0.0],'SALE PRICE_BUILDING_CLASS':[0.0]}
        df3 = pd.DataFrame(df2)

        std_encoding=df.groupby('borough').agg({'sale_price':['std']}).reset_index()
        std_encoding.columns = ['borough','sale_price_borough']
        sale_price_b = std_encoding['sale_price_borough'][std_encoding['borough']==df3['SALE PRICE_B'][0]]
        df3['SALE PRICE_BOROUGH'][0] = sale_price_b
        df3 = df3.drop('SALE PRICE_B',axis = 1)



        std_encoding=df.groupby('neighborhood').agg({'sale_price':['std']}).reset_index()
        std_encoding.columns = ['neighborhood','sale_price_neighborhood']
        sale_price_n = std_encoding['sale_price_neighborhood'][std_encoding['neighborhood']==df3['SALE PRICE_N'][0]]
        df3['SALE PRICE_NEIGHBORHOOD'][0] = sale_price_n
        df3 = df3.drop('SALE PRICE_N',axis = 1)



        std_encoding=df.groupby('tax_class').agg({'sale_price':['std']}).reset_index()
        std_encoding.columns = ['tax_class','sale_price_tax']
        if tax == 1:
            sale_price_t = std_encoding['sale_price_tax'][std_encoding['tax_class']==1]
            df3['SALE PRICE_TAX'][0] = sale_price_t
        elif tax == 2:
            sale_price_t = std_encoding['sale_price_tax'][std_encoding['tax_class']==2]
            df3['SALE PRICE_TAX'][0] = sale_price_t
        else:
            sale_price_t = std_encoding['sale_price_tax'][std_encoding['tax_class']==4]
            df3['SALE PRICE_TAX'][0] = sale_price_t

        df3 = df3.drop('SALE PRICE_T',axis = 1)


        std_encoding=df.groupby('building_class_category').agg({'sale_price':['std']}).reset_index()
        std_encoding.columns = ['building_class_category','sale_price_building_class']
        sale_price_bu = std_encoding['sale_price_building_class'][std_encoding['building_class_category']==df3['SALE PRICE_BG_CLASS'][0]]
        df3['SALE PRICE_BUILDING_CLASS'][0] = sale_price_bu
        df3 = df3.drop('SALE PRICE_BG_CLASS',axis = 1)

        df3['RESIDENTIAL UNITS'] = df3['RESIDENTIAL UNITS'].astype(float)
        df3['COMMERCIAL UNITS'] = df3['COMMERCIAL UNITS'].astype(float)
        df3['TOTAL UNITS'] = df3['TOTAL UNITS'].astype(float)
        df3['LAND SQUARE FEET'] = df3['LAND SQUARE FEET'].astype(float)
        df3['GROSS SQUARE FEET'] = df3['GROSS SQUARE FEET'].astype(float)
        df3['PROPERTY AGE'] = df3['PROPERTY AGE'].astype(float)
        df3['SALE PRICE_BOROUGH'] = df3['SALE PRICE_BOROUGH'].astype(float)
        df3['SALE PRICE_NEIGHBORHOOD'] = df3['SALE PRICE_NEIGHBORHOOD'].astype(float)
        df3['SALE PRICE_TAX'] = df3['SALE PRICE_TAX'].astype(float)
        df3['SALE PRICE_BUILDING_CLASS'] = df3['SALE PRICE_BUILDING_CLASS'].astype(float)

        df3_shap = shap_exp.shap_values(df3[0:1])

        predict_cluster = kmeans.predict(df3_shap)

        cluster1.to_csv('cluster1.csv')
        cluster2.to_csv('cluster2.csv')
        cluster3.to_csv('cluster3.csv')

        if predict_cluster[0] == 0:
            sample_url = "cluster1.csv"
        elif predict_cluster[1] == 1:
            sample_url = "cluster2.csv"
        else:
            sample_url = "cluster3.csv"

        st.title("Interactive Map Visualization")

        #sample_url = "cluster1.csv"
        url = st.text_input("Enter URL:", sample_url)
        m = leafmap.Map(locate_control=True, plugin_LatLngPopup=False)
        if url:

            try:
                df = pd.read_csv(url)

                columns = df.columns.values.tolist()
                row1_col1, row1_col2, row1_col3, row1_col4, row1_col5 = st.columns(
                    [1, 1, 3, 1, 1]
                )

                lon_index = 0
                lat_index = 0

                for col in columns:
                    if col.lower() in ["lon", "longitude", "long", "lng"]:
                        lon_index = columns.index(col)
                    elif col.lower() in ["lat", "latitude"]:
                        lat_index = columns.index(col)

                with row1_col1:
                    x = st.selectbox("Select longitude column", columns, lon_index)

                with row1_col2:
                    y = st.selectbox("Select latitude column", columns, lat_index)

                with row1_col3:
                    popups = st.multiselect("Select popup columns", columns, columns)

                with row1_col4:
                    heatmap = st.checkbox("Add heatmap")

                if heatmap:
                    with row1_col5:
                        if "pop_max" in columns:
                            index = columns.index("pop_max")
                        else:
                            index = 0
                            heatmap_col = st.selectbox("Select heatmap column", columns, index)
                        try:
                            m.add_heatmap(df, y, x, heatmap_col)
                        except:
                            st.error("Please select a numeric column")

            try:
                m.add_points_from_xy(df, x, y, popups)
            except:
                st.error("Please select a numeric column")

        except Exception as e:
            st.error(e)

        m.to_streamlit()
