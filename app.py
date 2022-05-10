import numpy as np 
import pandas as pd 
import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import streamlit as st
from PIL import Image

st.set_option('deprecation.showPyplotGlobalUse', False)

# Load  model a 
model = joblib.load(open("uplift_model.joblib","rb"))

def data_preprocessor(df):
    """this function preprocess the user input
        return type: pandas dataframe
    """
    #df.wine_type = df.wine_type.map({'white':0, 'red':1})
    #return df
    pass

def visualize_confidence_level(prediction):
    """
    this function uses matplotlib to create inference bar chart rendered with streamlit in real-time 
    return type : matplotlib bar chart  
    """
    data = (prediction[0]*100).round(2)
    #grad_percentage = pd.DataFrame(data = data,columns = ['Percentage'],index = ['Low','Ave','High'])
    grad_percentage = pd.DataFrame(data = data,columns = ['Percentage'],index = ['Chances'])
    ax = grad_percentage.plot(kind='barh', figsize=(7, 4), color='#722f37', zorder=10, width=0.5)
    ax.legend().set_visible(False)
    ax.set_xlim(xmin=0, xmax=100)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")
    
    vals = ax.get_xticks()
    for tick in vals:
        ax.axvline(x=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

    ax.set_xlabel("Winning Chances", labelpad=2, weight='bold', size=12)
    ax.set_ylabel("Zobia", labelpad=10, weight='bold', size=12)
    ax.set_title('Zobia Winning Chances ', fontdict=None, loc='center', pad=None, weight='bold')

    st.pyplot()
    return

st.write("""
# Zobia Winning Chances 
Toggle side-bar panel to observe chances of **Zobia** winning 
""")

#read in wine image and render with streamlit
#image = Image.open('wine_image.png')
#st.image(image, caption='wine company',use_column_width=True)

#st.sidebar.header('User Input Parameters') #user input parameter collection with streamlit side bar


def get_user_input():
    """
    this function is used to get user input using sidebar slider and selectbox 
    return type : pandas dataframe
    """
    f0  = st.sidebar.slider("Confidence", 0, 40, 20)
    f1  = st.sidebar.slider('Focus', 0, 20, 10)
    f2  = st.sidebar.slider('Hustle', 0, 20, 10)
    f3  = st.sidebar.slider('Patience', -20, 20, 0)
    f4  = st.sidebar.slider('Persistence', 0, 40, 10)
    f5  = st.sidebar.slider('Science', -20, 20, 0)
    f6  = st.sidebar.slider('General Knowledge', -40, 20, 0)
    f7  = st.sidebar.slider('Resilience', 0, 20, 10)
    f8  = st.sidebar.slider('Sports', 0, 20, 5)
    f9  = st.sidebar.slider('Mathematics', 0, 80, 20)
    f10 = st.sidebar.slider('Lethargy', 0, 20, 10)
    f11 = st.sidebar.slider('Distractions', -10, 20, 0)

    features = {'f0': f0,
            'f1': f1,
            'f2': f2,
            'f3': f3,
            'f4': f4,
            'f5': f5,
            'f6': f6,
            'f7': f7,
            'f8': f8,
            'f9': f9,
            'f10': f10,
            'f11': f11
            }
    data = pd.DataFrame(features,index=[0])

    return data

user_input_df = get_user_input()
processed_user_input = data_preprocessor(user_input_df)

st.subheader('User Input parameters')
st.write(user_input_df)

#prediction = model.predict(processed_user_input)
prediction = model.predict(user_input_df)
#prediction_proba = model.predict_proba(processed_user_input)

visualize_confidence_level(prediction)
