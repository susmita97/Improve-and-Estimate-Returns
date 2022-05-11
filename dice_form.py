import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from PIL import Image
import matplotlib.pyplot as plt

import dice_ml
from dice_ml import Dice

st.set_page_config(layout="wide")

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_imp():

    listfeat = ['RESIDENTIAL UNITS',
     'COMMERCIAL UNITS',
     'TOTAL UNITS',
     'LAND SQUARE FEET',
     'GROSS SQUARE FEET',
     'PROPERTY AGE',
     'SALE PRICE_BOROUGH',
     'SALE PRICE_NEIGHBORHOOD',
     'SALE PRICE_TAX',
     'SALE PRICE_BUILDING_CLASS']
    df_mergedfinal = pd.read_csv('df_mergedfinal.csv')
    train_X = pd.read_csv('trainx.csv')
    train_Y = pd.read_csv('trainy.csv')

    df = pd.read_csv('dfcopy.csv')

    df3 = train_X[0:1]

    rf_model = RandomForestRegressor()
    rf_model = rf_model.fit(train_X, train_Y.values.ravel())

    dice_data = dice_ml.Data(dataframe=df_mergedfinal, continuous_features=listfeat, outcome_name='SALE PRICE')

    model_dice = dice_ml.Model(model=rf_model, backend="sklearn", model_type='regressor')

    exp_genetic_boston = Dice(dice_data, model_dice, method="kdtree")

    return [exp_genetic_boston, rf_model, df, df3]

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

    listofelements = load_imp()

    exp_genetic_boston = listofelements[0]
    rf_model = listofelements[1]
    df = listofelements[2]
    df3 = listofelements[3]

    prediction = rf_model.predict(df3)

    st.title('Counterfactuals')

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

    st.write('Enter your future price of the property you want to have')
    st.write('Please enter both the values')

    range1 = form.number_input(label="Select a range(low)",min_value=prediction[0])
    range2 = form.number_input(label="Select a range(high)",min_value=prediction[0])
    submit_button = form.form_submit_button(label='Submit')

    if submit_button and range2>range1:

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

        query_instances_boston = df3
        genetic_boston = exp_genetic_boston.generate_counterfactuals(query_instances_boston,
                                                                     total_CFs=20,
                                                                     desired_range=[range1, range2],
        permitted_range={'TOTAL UNITS': [1, 60], 'COMMERCIAL UNITS': [1, 60], 'RESIDENTIAL UNITS': [1, 60],
                         'LAND SQUARE FEET': [1000,10000],'GROSS SQUARE FEET': [1000,10000]})

        cfdf = genetic_boston.cf_examples_list[0].final_cfs_df

        cfdf = cfdf.drop(['SALE PRICE_BOROUGH', 'SALE PRICE_NEIGHBORHOOD', 'SALE PRICE_TAX', 'SALE PRICE_BUILDING_CLASS'],axis=1)
        cfdf.reset_index(inplace=True)
        cfdf = cfdf.drop('index',axis=1)
        #st.write(cfdf)
        st.write('Your property attributes currently are')
        st.write(df3)
        st.write('The generated counterfactuals')
        st.write(cfdf)

        st.write('The counterfactuals show the attributes that are possibly changeable')
        st.write('Since the location of a property cannot be changed and it is unlikely that the property tax class or building category would change')
        st.write('Following shows the results in a visual way')

        s1 = pd.Series(cfdf['RESIDENTIAL UNITS'])
        s2 = pd.Series(df3.iloc[0]['RESIDENTIAL UNITS'])
        s3 = s1. append(s2,ignore_index=True)

        colors = []
        ylabels = []
        xvalues = []
        for i in range(0,len(s3)):
            colors.append('b')
            ylabels.append('CF'+str(i))
            xvalues.append(s3.iloc[i])

        colors[-1] = 'r'
        ylabels[-1] = 'OG'
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.bar(ylabels,xvalues,color=colors[-1],label='Original Value')
        ax.bar(ylabels,xvalues,color=colors,label='Counterfactual')
        ax.set_ylim(0,s3.max()+1)
        ax.legend()
        plt.show()

        st.pyplot(plt,bbox_inches='tight')

        s1 = pd.Series(cfdf['COMMERCIAL UNITS'])
        s2 = pd.Series(df3.iloc[0]['COMMERCIAL UNITS'])
        s3 = s1. append(s2,ignore_index=True)

        colors = []
        ylabels = []
        xvalues = []
        for i in range(0,len(s3)):
            colors.append('b')
            ylabels.append('CF'+str(i))
            xvalues.append(s3.iloc[i])

        colors[-1] = 'r'
        ylabels[-1] = 'OG'
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.bar(ylabels,xvalues,color=colors[-1],label='Original Value')
        ax.bar(ylabels,xvalues,color=colors,label='Counterfactual')
        ax.set_ylim(0,s3.max()+1)
        ax.legend()
        plt.show()

        st.pyplot(plt,bbox_inches='tight')

        s1 = pd.Series(cfdf['LAND SQUARE FEET'])
        s2 = pd.Series(df3.iloc[0]['LAND SQUARE FEET'])
        s3 = s1. append(s2,ignore_index=True)

        colors = []
        ylabels = []
        xvalues = []
        for i in range(0,len(s3)):
            colors.append('b')
            ylabels.append('CF'+str(i))
            xvalues.append(s3.iloc[i])

            colors[-1] = 'r'
            ylabels[-1] = 'OG'
            fig = plt.figure()
            ax = fig.add_axes([0,0,1,1])
            ax.bar(ylabels,xvalues,color=colors[-1],label='Original Value')
            ax.bar(ylabels,xvalues,color=colors,label='Counterfactual')
            ax.set_ylim(0,s3.max()+100)
            ax.legend()
            plt.show()

        st.pyplot(plt,bbox_inches='tight')

        s1 = pd.Series(cfdf['GROSS SQUARE FEET'])
        s2 = pd.Series(df3.iloc[0]['GROSS SQUARE FEET'])
        s3 = s1. append(s2,ignore_index=True)

        colors = []
        ylabels = []
        xvalues = []
        for i in range(0,len(s3)):
            colors.append('b')
            ylabels.append('CF'+str(i))
            xvalues.append(s3.iloc[i])

            colors[-1] = 'r'
            ylabels[-1] = 'OG'
            fig = plt.figure()
            ax = fig.add_axes([0,0,1,1])
            ax.bar(ylabels,xvalues,color=colors[-1],label='Original Value')
            ax.bar(ylabels,xvalues,color=colors,label='Counterfactual')
            ax.set_ylim(0,s3.max()+100)
            ax.legend()
            plt.show()

        st.pyplot(plt,bbox_inches='tight')

        s1 = pd.Series(cfdf['PROPERTY AGE'])
        s2 = pd.Series(df3.iloc[0]['PROPERTY AGE'])
        s3 = s1. append(s2,ignore_index=True)

        colors = []
        ylabels = []
        xvalues = []
        for i in range(0,len(s3)):
            colors.append('b')
            ylabels.append('CF'+str(i))
            xvalues.append(s3.iloc[i])

        colors[-1] = 'r'
        ylabels[-1] = 'OG'
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.bar(ylabels,xvalues,color=colors[-1],label='Original Value')
        ax.bar(ylabels,xvalues,color=colors,label='Counterfactual')
        ax.set_ylim(0,s3.max()+50)
        ax.legend()
        plt.show()

        st.pyplot(plt,bbox_inches='tight')
