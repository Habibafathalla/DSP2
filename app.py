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

selsect_col,graph=st.columns((1,4))



with selsect_col:
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
    flag=1
if Mode_Selection=='Musical Instruments' :
    sliders_number = 3

    lst_Drums=[0,500]
    lst_flute =[800,1500]
    lst_Xy=[500,800]
    lst_final=[lst_Drums,lst_flute,lst_Xy]

    flag=1
      
if Mode_Selection=='Vowels' :
    sliders_number = 3

    lst_z=[0,2400]
    lst_o=[0,890]
    lst_a=[0,2656]
    lst_e=[0,501]
    lst_final=[lst_o,lst_z,lst_a,lst_e]

    flag=1

if Mode_Selection=='Voice Changer' :
    sliders_number = 1  
      
if  flag==1:
    with selsect_col:
            upload_file= st.file_uploader("Upload your File",type='wav')
    
    if upload_file:

            st.session_state['audio'],st.session_state['sampleRare']= librosa.load(upload_file)
            audio_trim,_ = librosa.effects.trim(st.session_state['audio'], top_db=30)
            start_btn  =button("▷")
            pause_btn  =button(label='Pause')
            resume_btn =button(label='resume')
            st.session_state['audio']=audio_trim

        #play audio
            st.audio(upload_file, format='audio/wav')
        # transform to fourier 
            list_freq_domain,frequncies, magnitude,phase, number_samples = Functions.fourier_transformation(st.session_state['audio'], st.session_state['sampleRare'])
            freq_axis_list, amplitude_axis_list,bin_max_frequency_value = Functions.bins_separation(frequncies, magnitude, sliders_number)
            valueSlider = Functions.Sliders_generation(sliders_number)
            if Mode_Selection=="Uniform Range":
               Modified_signal=Functions.frequencyFunction(valueSlider, amplitude_axis_list) 
            elif Mode_Selection=="Vowels" or "Musical Instruments" :
               Modified_signal=Functions.final_func(list_freq_domain,frequncies,lst_final,valueSlider)
            else:
                 Modified_signal=magnitude
            st.session_state['fft_frequency']= np.abs(fft.rfftfreq(len(st.session_state['audio']),1/st.session_state['sampleRare']))
            fig_trans=px.line(x=st.session_state['fft_frequency'], y=st.session_state['spectrum']).update_layout(yaxis_title='Amp',xaxis_title='HZ')
            fig_spect =go.Figure(data =
            go.Heatmap(x = st.session_state['fft_frequency'], y= st.session_state['spectrum']))
            if Mode_Selection=="Uniform Range":
                  st.session_state['spectrum_inv']=Functions.inverse(Modified_signal,phase) 
            else:
                  st.session_state['spectrum_inv']=np.fft.irfft(Modified_signal) 
            #convert to audio
            result_bytes = Functions.convertToAudio(st.session_state['sampleRare'], st.session_state['spectrum_inv'])
            with selsect_col:
                st.audio(result_bytes, format='audio/wav')
            with graph:
                Functions.plotShow(st.session_state['audio'], st.session_state['spectrum_inv'], start_btn,pause_btn,resume_btn,valueSlider,st.session_state['sampleRare'])
                #ranges
                # fig_2= px.line(x=st.session_state['fft_frequency'], y=Modified_signal)
                # st.plotly_chart(fig_2,use_container_width=True)




    if Mode_Selection=='Voice Changer':
        with selsect_col:
            upload_file= st.file_uploader("Upload your File",type='wav')
        if upload_file:

            st.session_state['audio'],st.session_state['sampleRare']= librosa.load(upload_file)
            audio_trim,_ = librosa.effects.trim(st.session_state['audio'], top_db=30)
            st.session_state['audio']=audio_trim
        #play audio
            with selsect_col:
             st.audio(upload_file, format='audio/wav')
            # draw on time domain 
            t=np.array(range(0,len(st.session_state['audio'])))/st.session_state['sampleRare']
            # fig_trans=px.line(x=st.session_state['fft_frequency'], y=st.session_state['spectrum']).update_layout(yaxis_title='Amp',xaxis_title='HZ')
            fig_spect =go.Figure(data =
            go.Heatmap(x = st.session_state['fft_frequency'], y= st.session_state['spectrum']))
            st.session_state['spectrum_inv']= librosa.effects.pitch_shift(st.session_state['audio'] , sr= st.session_state['sampleRare'] , n_steps=st.session_state['sliderValues'][0][1])

            #convert to audio
            result_bytes = Functions.convertToAudio(st.session_state['sampleRare'], st.session_state['spectrum_inv'])
    
            with selsect_col:
             st.audio(result_bytes, format='audio/wav')
            
            with graph:
                  Functions.plotFunc(t, st.session_state['audio'], st.session_state['spectrum_inv'])
    

           
       


