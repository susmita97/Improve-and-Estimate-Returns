import streamlit as st
from multiapp import MultiApp
import form, dice_form, leafmap_shap

app = MultiApp()

st.markdown("""
# Improve and Estimate Real Estate Returns
The application helps you to predict the current and future price for your property, and generate an explanation for the 
feature affecting the price (Select Option One). 
It will also generate and tell you values for the features below required to attain a particular sale price point you want
(Select Option Two). 
Find similar properties which have the same features affecting the price point (Select Option Three).
""")

# Add all your application here
app.add_app("Predict current or future price", form.app)
app.add_app("Get Counterfactuals", dice_form.app)
app.add_app("Find similar properties", leafmap_shap.app)
# The main app
app.run()
