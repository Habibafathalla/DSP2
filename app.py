import streamlit as st
import streamlit.components.v1 as com
from streamlit_option_menu import option_menu
from streamlit import button
import streamlit as st
import  streamlit_vertical_slider  as svs
import numpy as np
import scipy as sc
from scipy.interpolate import interp1d, interp2d,splev
import pandas as pd
from math import ceil,floor
import plotly.express as px
from plotly.express import colors
import matplotlib.pyplot as plt
from glob import glob
import librosa 
import librosa.display
import IPython.display as ipd
import matplotlib.pylab as plt 


st.set_page_config(page_title="equalizer", page_icon=":bar_chart:",layout="wide")


#hiding copyright things
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """

# Initialization of session state
if 'sliderValues' not in st.session_state:
    st.session_state['sliderValues'] = []

if 'groups' not in st.session_state:
    st.session_state['groups'] = []
if 'audio' not in st.session_state:
    st.session_state['audio'] = []
if 'sampleRare' not in st.session_state:
    st.session_state['sampleRare'] = []

def slider_group(groups):
    adjusted_data = []
    sliders = {}
    columns = st.columns(len(groups))
    for idx, i in enumerate(groups):
        min_value = i[0]
        max_value = i[1]
        key = f'member{str(idx)}'
        with columns[idx]:
            sliders[f'slider_group_{key}'] = svs.vertical_slider(key=key, default_value=i[2], step=1, min_value=min_value, max_value=max_value)
            if sliders[f'slider_group_{key}'] == None:
                sliders[f'slider_group_{key}']  = i[2]
            adjusted_data.append((i[0],i[1],sliders[f'slider_group_{key}'] ))
    return adjusted_data        


st.session_state['groups'] = [(0,200,100),
            (0,200,150),
            (0,200,75),
            (0,200,25),
            (0,200,150),
            (0,200,60),
            (0,200,86),
            (0,200,150),
            (0,200,150),
            (0,200,25)]

upload_file= st.file_uploader("")
if upload_file:
    st.session_state['audio'],st.session_state['sampleRare']=librosa.load(upload_file)
    #play audio
    st.audio(upload_file, format='audio/wav')
    fig=px.line((st.session_state['audio']))
    st.plotly_chart(fig, use_container_width=True)



st.session_state['sliderValues']=slider_group(st.session_state['groups'])
# st.write(st.session_state)
