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
import matplotlib.pyplot as plt
from glob import glob
import librosa 
import librosa.display
from numpy import fft
import xlsxwriter
import io
from io import BytesIO 
from scipy.io.wavfile import write
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import time as ti
from scipy import signal
from scipy.io import loadmat
from Functions import Functions


st.set_page_config(page_title="Equalizer", page_icon=":bar_chart:",layout="wide")

st.set_option('deprecation.showPyplotGlobalUse', False)

reduce_header_height_style = """
    <style>
        div.block-container {padding-top:0rem;}
        header {visibility: hidden;}
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
"""
st.markdown(reduce_header_height_style, unsafe_allow_html=True)


st.write(
    '<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: left;} </style>', unsafe_allow_html=True)


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


local_css("style.css")



# Initialization of session state
if 'sliderValues' not in st.session_state:
    st.session_state['sliderValues'] = []

if 'groups' not in st.session_state:
    st.session_state['groups'] = []
if 'audio' not in st.session_state:
    st.session_state['audio'] =[]
if 'sampleRare' not in st.session_state:
    st.session_state['sampleRare'] = []
if 'spectrum' not in st.session_state:
    st.session_state['spectrum'] = []
if 'fft_frequency' not in st.session_state:
    st.session_state['fft_frequency'] = []

if  'Uniform_Range_Default' not in st.session_state:
    st.session_state['Uniform_Range_Default']=[]
if  'spectrum_inv' not in st.session_state:
    st.session_state['spectrum_inv']=[]


if  'spectrum_inv_uniform' not in st.session_state:
    st.session_state['spectrum_inv_uniform']=[]

if 'start' not in st.session_state:
    st.session_state['start']=0
if 'size1' not in st.session_state:
    st.session_state['size1']=0
if 'flag' not in st.session_state:
    st.session_state['flag'] = 0

select_col,graph=st.columns((1,4))



with select_col:
     Mode_Selection=st.selectbox(
     'Equalizer',
     ('Uniform Range', 'Vowels', 'Musical Instruments','Voice Changer'))
     spec_visibality=st.radio(
     "Spectogram",
     ('Hide', 'Show'))
     if spec_visibality=='Show':
        Functions.plotSpectrogram(st.session_state['audio'],st.session_state['sampleRare'],'Input')
        Functions.plotSpectrogram(st.session_state['spectrum_inv'],st.session_state['sampleRare'],'Output')


if Mode_Selection=='Uniform Range' :
    sliders_number = 10
    text=["1st","2nd","3rd","4th","5th","6th","7th","8th","9th","10th"]
    flag=1
if Mode_Selection=='Musical Instruments' :
    sliders_number = 3

    lst_Drums=[(0,500)]
    lst_flute =[(800,1500)]
    lst_Xy=[(500,800)]
    lst_final_music=[lst_Drums,lst_flute,lst_Xy]
    text=["Drums","Flute","Xy"]

    flag=1
      
if Mode_Selection=='Vowels' :
    sliders_number = 4

    lst_z=[(0,2400)]
    lst_o=[(0,890)]
    lst_a=[(0,2656)]
    lst_e=[(82,182),(214,314),(2039,2639),(3156,3756)]
    lst_final=[lst_o,lst_z,lst_a,lst_e]
    text=["A","C","M","E"]

    flag=1

if Mode_Selection=='Voice Changer' :
    sliders_number = 1 
    flag= 1
    text=["Female to male"]
      
with select_col:
        upload_file= st.file_uploader("Upload your File",type='wav')
if not upload_file:
    st.session_state['audio'],st.session_state['sampleRare']=librosa.load("audio\hello-female-friendly-professional.wav")
else:
    st.session_state['audio'],st.session_state['sampleRare']= librosa.load(upload_file)
if  flag==1:
    audio_trim,_ = librosa.effects.trim(st.session_state['audio'], top_db=30)
    st.session_state['audio']=audio_trim
    #play audio
    with select_col:
        if upload_file:
            st.audio(upload_file, format='audio/wav')
        else:
            input_bytes=Functions.convertToAudio(st.session_state['sampleRare'],st.session_state['audio'])
            st.audio(input_bytes, format='audio/wav')

    
    c0,c1,c2,c3,c4,c5,c6=st.columns((1,2,3,4,5,6,7),gap="small")
    with c4:   
        start_btn  =button("Play")
    with c5: 
        pause_btn  =button(label='Pause')
    with c6: 
        resume_btn =button(label='Resume')

    
    # transform to fourier 
    spectrum,frequencies, magnitude,phase, number_samples = Functions.fourier_transformation(st.session_state['audio'], st.session_state['sampleRare'])
    freq_axis_list, amplitude_axis_list,bin_max_frequency_value = Functions.bins_separation(frequencies, magnitude, sliders_number)
    valueSlider = Functions.Sliders_generation(sliders_number,text)
    value=valueSlider[0]
    if Mode_Selection=="Voice Changer":
       st.session_state['spectrum_inv']= librosa.effects.pitch_shift(st.session_state['audio'] , sr= st.session_state['sampleRare'] , n_steps=value)
       
        #convert to audio
    
       result_bytes = Functions.convertToAudio(st.session_state['sampleRare'], st.session_state['spectrum_inv'])
       with select_col:
          st.audio(result_bytes, format='audio/wav')
    
       with graph:
          Functions.plotShow(st.session_state['audio'], st.session_state['spectrum_inv'], start_btn,pause_btn,resume_btn,valueSlider,st.session_state['sampleRare'])
        #   Functions.plotShow(Time, st.session_state['audio'], st.session_state['spectrum_inv'])
    else:  
        if Mode_Selection=="Uniform Range":
          Modified_signal=Functions.frequencyFunction(valueSlider, amplitude_axis_list) 
        elif Mode_Selection=="Vowels" :
          Modified_signal=Functions.modify_magnitude(spectrum,frequencies,lst_final,valueSlider)
        elif Mode_Selection=="Musical Instruments":
          Modified_signal=Functions.modify_magnitude(spectrum,frequencies,lst_final_music,valueSlider)

        else:
            Modified_signal=magnitude

        st.session_state['fft_frequency']= np.abs(fft.rfftfreq(len(st.session_state['audio']),1/st.session_state['sampleRare']))
        fig_trans=px.line(x=st.session_state['fft_frequency'], y=st.session_state['spectrum']).update_layout(yaxis_title='Amp',xaxis_title='HZ')
        if Mode_Selection=="Uniform Range":
            st.session_state['spectrum_inv']=Functions.inverse(Modified_signal,phase) 
        elif Mode_Selection== "Vowels" or" Musical Instruments":
            st.session_state['spectrum_inv']=np.fft.irfft(Modified_signal)

        elif Mode_Selection=="Voice Changer":
            flag==1
            st.session_state['spectrum_inv']=librosa.effects.pitch_shift(st.session_state['audio'] , sr= st.session_state['sampleRare'] , n_steps=valueSlider)
        # else:
        #     st.session_state['spectrum_inv']=np.fft.irfft(Modified_signal) 
        #convert to audio
        result_bytes = Functions.convertToAudio(st.session_state['sampleRare'], st.session_state['spectrum_inv'])
        with graph:
            Functions.plotShow(st.session_state['audio'], st.session_state['spectrum_inv'], start_btn,pause_btn,resume_btn,valueSlider,st.session_state['sampleRare'])
            #ranges
            # fig_2= px.line(x=st.session_state['fft_frequency'], y=Modified_signal)
            # st.plotly_chart(fig_2,use_container_width=True)
        with select_col:
            st.audio(result_bytes, format='audio/wav')
    

           
       


