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
from numpy import fft
import xlsxwriter
import io
from io import BytesIO 
from scipy.signal import find_peaks
from scipy.io.wavfile import write
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import time as ti
from scipy import signal
from scipy.io import loadmat
import plotly.express as px
import matplotlib.animation as animation
import altair as alt
from pandas import Series
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

if 'start' not in st.session_state:
    st.session_state['start']=0
if 'size1' not in st.session_state:
    st.session_state['size1']=0
if 'flag' not in st.session_state:
    st.session_state['flag'] = 0

 
time = np.linspace(0,1,1000)

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
            st.text(i[4])
            if sliders[f'slider_group_{key}'] == None:
                sliders[f'slider_group_{key}']  = i[2]
            adjusted_data.append((i[3],(sliders[f'slider_group_{key}'] )))
    return adjusted_data


selsect_col,graph=st.columns((1,4))

lst_z=[0,2400]
lst_o=[0,890]
lst_a=[0,2656]
lst_e=[0,501]
    
lst_final=[lst_o,lst_z,lst_a,lst_e]

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




      
if Mode_Selection=='Vowels' or Mode_Selection=='Musical Instruments' :
    sliders_number = 3

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

        # with selsect_col:
        st.audio(upload_file, format='audio/wav')

    # draw on time domain 
        t=np.array(range(0,len(st.session_state['audio'])))/st.session_state['sampleRare']

    # transform to fourier 
        list_freq_domain,frequncies, magnitude,phase, number_samples = Functions.fourier_transformation(st.session_state['audio'], st.session_state['sampleRare'])
        freq_axis_list, amplitude_axis_list,bin_max_frequency_value = Functions.bins_separation(frequncies, magnitude, sliders_number)
        valueSlider = Functions.Sliders_generation(sliders_number)
        Modified_signal=Functions.final_func(list_freq_domain,frequncies,lst_final,valueSlider)
       
        #control amp
        
        st.session_state['fft_frequency']= np.abs(fft.rfftfreq(len(st.session_state['audio']),1/st.session_state['sampleRare']))

        fig_trans=px.line(x=st.session_state['fft_frequency'], y=st.session_state['spectrum']).update_layout(yaxis_title='Amp',xaxis_title='HZ')
        fig_spect =go.Figure(data =
        go.Heatmap(x = st.session_state['fft_frequency'], y= st.session_state['spectrum']))
        st.session_state['spectrum_inv']=np.fft.irfft(Modified_signal) 

        #convert to audio
        result_bytes = Functions.convertToAudio(st.session_state['sampleRare'], st.session_state['spectrum_inv'])


        with selsect_col:
            st.audio(result_bytes, format='audio/wav')
        with graph:


        #    Functions.plotFunc(t, st.session_state['audio'], st.session_state['spectrum_inv'],start_btn,resume_btn,pause_btn)
            Functions.plotShow(st.session_state['audio'], st.session_state['spectrum_inv'], start_btn,pause_btn,resume_btn,valueSlider,st.session_state['sampleRare'])



if Mode_Selection=='Uniform Range':

        with selsect_col:
           upload_file= st.file_uploader("Upload your File",type='csv')
        
        if upload_file:
            csv_data = pd.read_csv(upload_file)
            csv_data=csv_data.to_numpy()
            time=csv_data[:,0]
            st.session_state['Uniform_Range_Default']=csv_data[:,1]
        else:
            st.session_state['Uniform_Range_Default']=4*np.sin(2*np.pi*2*time)

        signal=fft.fft(st.session_state['Uniform_Range_Default'])
        Fs = len(st.session_state['Uniform_Range_Default'])/10
        T = 1 / Fs
        spectrum=np.abs( signal)
        fft_frequency= abs(fft.fftfreq(len(st.session_state['Uniform_Range_Default']),T))
        fft_phase=np.angle(signal)

           
        fig_trans=px.line(x=fft_frequency, y=spectrum).update_layout(yaxis_title='Amp',xaxis_title='HZ')
        fig_spect =go.Figure(data =
        go.Heatmap(x = fft_frequency, y= spectrum))
        spectrum_inv =Functions.inverse(spectrum, fft_phase)
        with graph:
            # zoom_fuc(time,st.session_state['Uniform_Range_Default'],spectrum_inv)
            Functions.plotFunc(time, st.session_state['Uniform_Range_Default'], spectrum_inv)


    
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

        #  zoom_fuc(t,st.session_state['audio'],st.session_state['spectrum_inv'])
           Functions.plotFunc(t, st.session_state['audio'], st.session_state['spectrum_inv'])
    


