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
# import time
from scipy import signal
from scipy.io import loadmat
import plotly.express as px


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
            if factor!=0:
                if factor>0:
                    y[j]+=y[j]*(factor/20)
                elif factor<0:
                    y[j]=y[j]-y[j]*(abs(factor)/20)
    return y

def zoom_fuc(x,y,y_inverse):
    fig =  make_subplots(rows=1, cols=2,
                    shared_xaxes='all', shared_yaxes='all',
                    vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x = x, y = y), 1,1 )
    fig.add_trace(go.Scatter(x = x, y = y_inverse), 1,2 )
    return  st.plotly_chart(fig,use_container_width=True)

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

def plotSpectrogram(audioData, fs, Title):
    N = 512
    w = signal.blackman(N)
    freqs, time, Pxx = signal.spectrogram(audioData, fs, window=w, nfft=N)

    layout = go.Layout(margin=go.layout.Margin(l=0, r=0, b=0, t=30,))
    fig = go.Figure(layout=layout)

    fig.add_trace(go.Heatmap(x=time, y=freqs, z=10*np.log10(Pxx),
                  colorscale='Jet', name='Spectrogram'))

    fig.update_layout(height=300, title={
        'text': Title,
        'y': 1,
        'x': 0.49,
        'xanchor': 'center',
        'yanchor': 'top'},
        title_font=dict(
        family="Arial",
        size=17))
    fig.update_xaxes(title='Time')
    fig.update_yaxes(title='Frequency')
    st.sidebar.plotly_chart(fig, use_container_width=True)

ranges={
    'e':np.concatenate([[0,1],range(138,169),range(279,327),range(2095,2400)]),
    'o':np.concatenate([[0,1],range(138,152),range(271,304),range(392,455),range(561,600),range(700,891)]),
    'a':np.concatenate([[0,1],range(115,153),range(242,280),range(370,500),range(500,1200),range(1300,2656)]),
    'z':np.concatenate([range(0,6),range(70,210),range(250,501)])
    

}
st.session_state['groups'] = [(-20,20,0,ranges['e'],'E'),
            (-20,20,0,ranges['o'],'O'),
            (-20,20,0,ranges['a'],'A'),
            (-20,20,0,ranges['z'],'Z'),
            (-20,20,0,ranges['z'],'1st range'),
            (-20,20,0,ranges['z'],'Paino'),
            (-20,20,0,ranges['z'],'Z'),
            (-20,20,0,ranges['z'],'Z'),
            (-20,20,0,ranges['z'],'Z'),
            (-20,20,0,ranges['z'],'Z')]

selsect_col,graph=st.columns((1,4))



with selsect_col:
     Mode_Selection=st.selectbox(
     'Equalizer',
     ('Uniform Range', 'Vowels', 'Musical Instruments',"ECG Abnormalities",'Voice Changer'))
    #  st.write("")
     spec_visibality=st.radio(
     "Spectogram",
     ('Hide', 'Show'))
     if spec_visibality=='Show':
        plotSpectrogram(st.session_state['audio'],st.session_state['sampleRare'],'Input')
        plotSpectrogram(st.session_state['spectrum_inv'],st.session_state['sampleRare'],'Output')




      
if Mode_Selection=='Vowels' or Mode_Selection=='Musical Instruments' :
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

        # transform to fourier 
           signal=fft.fft(st.session_state['audio'])
           st.session_state['spectrum']=np.abs( signal)
           st.session_state['fft_frequency']= fft.fftfreq(len(st.session_state['audio']),1/st.session_state['sampleRare'])
           fft_phase=np.angle(signal)
           #control amp
           for i in st.session_state['sliderValues']:

        # freq_list=select_range(i[0],0,4000,True)
            st.session_state['spectrum']=change_amplitude(st.session_state['fft_frequency'],st.session_state['spectrum'], i[0], i[1])
           
           fig_trans=px.line(x=st.session_state['fft_frequency'], y=st.session_state['spectrum']).update_layout(yaxis_title='Amp',xaxis_title='HZ')
           fig_spect =go.Figure(data =
           go.Heatmap(x = st.session_state['fft_frequency'], y= st.session_state['spectrum']))
           st.session_state['spectrum_inv']=inverse(st.session_state['spectrum'], fft_phase) 

         #convert to audio
           result_bytes = convertToAudio(st.session_state['sampleRare'], st.session_state['spectrum_inv'])
 
           with selsect_col:
            st.audio(result_bytes, format='audio/wav')
        
           with graph:
            zoom_fuc(t,st.session_state['audio'],st.session_state['spectrum_inv'])



if Mode_Selection=='Uniform Range':
        with selsect_col:
           upload_file= st.file_uploader("Upload your File",type='csv')
        
        if upload_file:
            st.session_state['Uniform_Range_Default'] = pd.read_csv(upload_file)
        else:
            st.session_state['Uniform_Range_Default']=4*np.sin(2*np.pi*2*time)

        signal=fft.fft(st.session_state['Uniform_Range_Default'])
        Fs = len(st.session_state['Uniform_Range_Default'])/10
        T = 1 / Fs
        spectrum=np.abs( signal)
        fft_frequency= abs(fft.fftfreq(len(st.session_state['Uniform_Range_Default']),T))
        fft_phase=np.angle(signal)
        for i in st.session_state['sliderValues']:
             spectrum=change_amplitude(fft_frequency,spectrum, i[0], i[1])
           
        fig_trans=px.line(x=fft_frequency, y=spectrum).update_layout(yaxis_title='Amp',xaxis_title='HZ')
        fig_spect =go.Figure(data =
        go.Heatmap(x = fft_frequency, y= spectrum))
        spectrum_inv =inverse(spectrum, fft_phase)
        with graph:
            zoom_fuc(time,st.session_state['Uniform_Range_Default'],spectrum_inv)


if Mode_Selection=='ECG Abnormalities':
        with selsect_col:
            upload_file= st.file_uploader("Upload your File")
        if upload_file:
           Signal=loadmat(upload_file)
           st.session_state['ECG'] =(Signal["val"][0])/200   # 200 is a data gain  
           Samples = len(  st.session_state['ECG'])
           Fs = Samples/10
           T = 1 / Fs
           Time=np.linspace(0,Samples*T,Samples)
           signal=fft.fft(st.session_state['ECG']*.001)
           st.session_state['spectrum']=(np.abs( signal))
           st.session_state['fft_frequency']= np.abs(fft.fftfreq(Samples,T))
           signal_phase=np.angle(signal)
           st.session_state['spectrum_inv']=inverse(st.session_state['spectrum'],signal_phase) 
           with graph:
            zoom_fuc(Time,st.session_state['ECG'],st.session_state['spectrum_inv'])
    
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
        result_bytes = convertToAudio(st.session_state['sampleRare'], st.session_state['spectrum_inv'])
 
        with selsect_col:
         st.audio(result_bytes, format='audio/wav')
        
        with graph:
         zoom_fuc(t,st.session_state['audio'],st.session_state['spectrum_inv'])
    


st.session_state['sliderValues']=slider_group(st.session_state['groups'])
# download(st.session_state['spectrum'],st.session_state['fft_frequency'])
