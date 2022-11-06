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

st.set_page_config(page_title="Equalizer", page_icon=":bar_chart:",layout="wide")

st.set_option('deprecation.showPyplotGlobalUse', False)

reduce_header_height_style = """
    <style>
        div.block-container {padding-top:0rem;}
    </style>
"""
st.markdown(reduce_header_height_style, unsafe_allow_html=True)


st.write(
    '<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: left;} </style>', unsafe_allow_html=True)


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


local_css("style.css")




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
    st.session_state['audio'] =[]
if 'sampleRare' not in st.session_state:
    st.session_state['sampleRare'] = []
if 'spectrum' not in st.session_state:
    st.session_state['spectrum'] = []
if 'fft_frequency' not in st.session_state:
    st.session_state['fft_frequency'] = []




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
            adjusted_data.append((i[3],(sliders[f'slider_group_{key}'] )))
    return adjusted_data

def download(amp , HZ):
    output = BytesIO()
    # Write files to in-memory strings using BytesIO
    # See: https://xlsxwriter.readthedocs.io/workbook.html?highlight=BytesIO#constructor
    workbook = xlsxwriter.Workbook(output, {'in_memory': True})
    worksheet = workbook.add_worksheet()
    worksheet.write(0,0,"amp")
    worksheet.write(0,1,"freq")
    worksheet.write_column(1,0,amp)
    worksheet.write_column(1,1,HZ)

    workbook.close()
    #Button of downloading
    st.download_button(
        label="Download",
        data=output.getvalue(),
        file_name="signal.xlsx",
        mime="application/vnd.ms-excel"
    ) 

# def extract_peak_frequency(data, sampling_rate):
#     fft_data = np.fft.fft(data)
#     freqs = np.fft.fftfreq(len(data))
    
#     peak_coefficient = np.argmax(np.abs(fft_data))
#     peak_freq = freqs[peak_coefficient]
    
#     return abs(peak_freq * sampling_rate)

def change_amplitude(x,y,frequencies,factor):
    for i in range(len(frequencies)):
        j=np.where(np.round(x)==frequencies[i])
        if j:
            if factor>0:
                y[j]=y[j]*(factor/20)
            elif factor<0:
                y[j]=y[j]-y[j]*(abs(factor)/20)
    return y
def convertToAudio(sr,signal):
    bytes_wav = bytes()
    byte_io = io.BytesIO(bytes_wav)
    write(byte_io,sr, signal.astype(np.float32))
    result_bytes = byte_io.read()
    return result_bytes

def inverse(amp,phase):
    combined=np.multiply(amp,np.exp(1j*phase))
    inverse_combined=fft.ifft(combined)
    signal=np.real(inverse_combined)
    return signal

                


ranges={
    '0':np.concatenate([range(1,5),range(100,1000),range(1000,4001)]),
    '1':np.concatenate([range(50,201),range(2000,3001)])
}

st.session_state['groups'] = [(-20,20,0,ranges['0']),
            (-20,20,0,ranges['1']),
            (-20,20,0,ranges['1']),
            (-20,20,0,ranges['1']),
            (-20,20,0,ranges['1']),
            (-20,20,0,ranges['1']),
            (-20,20,0,ranges['1']),
            (-20,20,0,ranges['1']),
            (-20,20,0,ranges['1']),
            (-20,20,0,ranges['1'])]




selsect_col,input_col,output_col=st.columns((1,2,2))


with selsect_col:
     st.selectbox(
     'Equalizer',
     ('Uniform Range', 'Vowles', 'Music',"ECG"))
     upload_file= st.file_uploader("")
     st.write("")
     st.radio(
     "Spectogram",
     ('Show', 'Hide'))



# upload_file= st.file_uploader("")
if upload_file:
    st.session_state['audio'],st.session_state['sampleRare']= librosa.load(upload_file)
    #play audio
    with input_col:


        st.audio(upload_file, format='audio/wav')

    # audio_trim,_ = librosa.effects.trim(st.session_state['audio'], top_db=30)

    # draw on time domain 
        t=np.array(range(0,len(st.session_state['audio'])))/st.session_state['sampleRare']
        fig=px.line(x=t,y=st.session_state['audio']).update_layout(xaxis_title='time(sec)')
        st.plotly_chart(fig, use_container_width=True)


    # transform to fourier 
    signal=fft.fft(st.session_state['audio'])
    st.session_state['spectrum']=np.abs( signal)
    st.session_state['fft_frequency']= np.abs(fft.fftfreq(len(st.session_state['audio']),1/st.session_state['sampleRare']))
    fft_phase=np.angle(signal)
    #control amp
    for i in st.session_state['sliderValues']:
        st.session_state['spectrum']=change_amplitude(st.session_state['fft_frequency'], st.session_state['spectrum'],i[0], i[1])
    fig_trans=px.line(x=st.session_state['fft_frequency'], y=st.session_state['spectrum']).update_layout(yaxis_title='Amp',xaxis_title='HZ')
    
    spectrum_inv=inverse(st.session_state['spectrum'], fft_phase) 
    fig_inv=px.line(x=t,y=spectrum_inv).update_layout(xaxis_title='time(sec)')

    #convert to audio
    result_bytes = convertToAudio(st.session_state['sampleRare'], spectrum_inv)
    with output_col:
        st.audio(result_bytes, format='audio/wav')
        st.plotly_chart(fig_inv, use_container_width=True)


    # st.plotly_chart(fig, use_container_width=True)
    # st.plotly_chart(fig_inv, use_container_width=True)
    # st.plotly_chart(fig_trans, use_container_width=True)

st.session_state['sliderValues']=slider_group(st.session_state['groups'])
download(st.session_state['spectrum'],st.session_state['fft_frequency'])
