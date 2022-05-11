import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from PIL import Image
import matplotlib.pyplot as plt

import lime
import lime.lime_tabular
import shap

import dill

@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def load_imp():
    df = pd.read_csv('dfcopy.csv')

    train_X = pd.read_csv('trainx.csv')
    train_Y = pd.read_csv('trainy.csv')
    test_X = pd.read_csv('testx.csv')
    test_Y = pd.read_csv('testy.csv')

    featlist = list(train_X.columns)
    featlist.append('SALE PRICE')

    rf_model = RandomForestRegressor()
    rf_model = rf_model.fit(train_X, train_Y.values.ravel())

    explainer = lime.lime_tabular.LimeTabularExplainer(np.array(train_X),feature_names = featlist, verbose=True, mode='regression', random_state = 2022, categorical_features = [6,7,8,9])

    return [df, train_X, train_Y, featlist, rf_model, explainer,test_X,test_Y]



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

def app():

    st.title('Real Estate Analytics')
    st.write('Please enter some information below about your property')
    st.write('The system will predict the current price of your property')
    st.write('All are required fields')

    listofelements = load_imp()

    df = listofelements[0]
    train_X = listofelements[1]
    train_Y = listofelements[2]
    featlist = listofelements[3]
    rf_model = listofelements[4]
    explainer = listofelements[5]
    test_X = listofelements[6]
    test_Y = listofelements[7]

    option = st.selectbox('Borough',
         ('Brooklyn', 'Bronx', 'Manhattan','Queens','Staten Island'))

    listofn = display_neighborhoods(df)

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

    limebutton = 1

    if submit_button or limebutton:

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

        prediction = rf_model.predict(df3)

        st.write('The current price of your property is :',prediction)

        #st.write(df3)
        #st.write(df3.count())

        st.write('Lime Explanation for Individual Predictions')

        st.write('You can select the number of features or samples you want to consider for your local explanation')

        lime_form = st.form(key='lime_form')

        numfeat = lime_form.slider(label="Select Number of Features",key = "numfeat", min_value=3, max_value=10)

        numsamp = lime_form.slider(label="Select Number of Samples",key = "numsamp", min_value=100, max_value=1500)

        limebutton = lime_form.form_submit_button(label = "Change")

        explanation = explainer.explain_instance(df3.iloc[0], rf_model.predict, num_features= numfeat, num_samples=numsamp)
        st.pyplot(explanation.as_pyplot_figure(),bbox_inches='tight')

        features = train_X.columns
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)

        st.write('Global Explanation For Comparison')
        st.write('Below is a bar chart showing the Global Importance of features for the Random Forest model')
        plt.figure(figsize=(13,10))
        plt.title('Feature Importances - Random Forest')
        plt.barh(range(len(indices)), importances[indices], color='blue', align='center')
        plt.yticks(range(len(indices)), features[indices])
        plt.xlabel('Relative Importance')
        st.pyplot(plt,bbox_inches='tight')

        #lime_chart = components.html(explanation.as_html(), height=900, width=700)
        #lime_chart

        st.write('SHAP Explanation summary for the model')
        explainer = shap.TreeExplainer(rf_model)

        with open('shap_values', 'rb') as f:
            shap_values = dill.load(f)


        shap_values_input = explainer.shap_values(df3[0:1])

        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(shap.summary_plot(shap_values, test_X, plot_type="bar"))

        st.write('SHAP Explanation for the input')

        st.pyplot(shap.decision_plot(explainer.expected_value, shap_values_input, test_X))
