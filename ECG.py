import streamlit as st
import matplotlib.pylab as plt 
import numpy as np
from numpy import fft
from scipy.io import loadmat
import plotly.express as px


if 'ECG' not in st.session_state:
    st.session_state['ECG'] =[]
if 'spectrum' not in st.session_state:
    st.session_state['spectrum']=[]
if 'fft_frequency' not in st.session_state:
    st.session_state['fft_frequency'] = []    




def inverse(amp,phase):
    combined=np.multiply(amp,np.exp(1j*phase))
    inverse_combined=fft.ifft(combined)
    signal=np.real(inverse_combined)
    return signal

uploaded_file = st.file_uploader('upload', label_visibility="hidden")  
if uploaded_file is not None:
   Signal=loadmat(uploaded_file)
   st.session_state['ECG'] =(Signal["val"][0])/200   # 200 is a data gain  
   Samples = len(  st.session_state['ECG'])
   Fs = Samples/10
   T = 1 / Fs
   Time=np.linspace(0,Samples*T,Samples)
   #Time domain
   fig = px.line(x=Time, y=st.session_state['ECG'])
   st.plotly_chart(fig,use_container_width=True)

   #FFT
   signal=fft.fft(st.session_state['ECG']*.001)
   st.session_state['spectrum']=(np.abs( signal))
   st.session_state['fft_frequency']= np.abs(fft.fftfreq(Samples,T))
   signal_phase=np.angle(signal)

   # Freqency domain
   fig_2= px.line(x=st.session_state['fft_frequency'], y=st.session_state['spectrum'])
   st.plotly_chart(fig_2,use_container_width=True)

   
   #signal inverse 
   spectrum_inv=inverse(st.session_state['spectrum'],signal_phase) 
   fig_inv=px.line(x=Time,y=spectrum_inv).update_layout(xaxis_title='time(sec)')
   st.plotly_chart(fig_inv,use_container_width=True)
   
