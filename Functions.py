import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import math
import streamlit as st
import  streamlit_vertical_slider  as svs
import io
from io import BytesIO 
from scipy.io.wavfile import write
from scipy import signal
import altair as alt
import time
import pandas as pd
import altair as alt
import plotly.graph_objects as go




class Functions():
    def frequencyFunction(values, amplitude_axis_list):
        flist =[]
        for i in range(0, 10):
                flist.append(amplitude_axis_list[i] * (1+(values[i]/20)))
                
        flat_list =[]
        for sublist in flist:
                for item in sublist:
                    flat_list.append(item)

        return flat_list
    def final_func(fou_of_signal,frequencies,list_of_freqs,list_of_sliders):
        final_fou = fou_of_signal
        for iter in range(len(list_of_sliders)):
            freqs_update = Functions.select_range(frequencies,list_of_freqs[iter][0],list_of_freqs[iter][1],True)
            final_fou = Functions.modify_magnitude(freqs_update,fou_of_signal,list_of_sliders[iter])
        return final_fou
    def read_ecg_file(file_path):
        signal = loadmat(file_path)
        signal = signal['val'][0]
        return signal
    def modify_magnitude(freq_list,list_freq_domain,factor):
        if factor!=0:
            list_freq_domain[freq_list] = list_freq_domain[freq_list] * ((factor/20)+1)
        return list_freq_domain
    def fourier_transformation(time_domain_data, sampling_rate):
        frequencies = np.fft.rfftfreq(len(time_domain_data), 1/sampling_rate)
        freq_domain_data = np.fft.rfft(time_domain_data)
        phase = np.angle(freq_domain_data)
        magnitude = np.abs(freq_domain_data)
        return freq_domain_data, frequencies, magnitude, phase, len(time_domain_data)
    def select_range(frequencies,min,max,equal):
        if equal:
            selected_freqs = (frequencies>=min)&(frequencies<=max)
        else:
            selected_freqs = (frequencies>min)&(frequencies<max)
        return 

    def bins_separation(frequency, amplitude,sliders_number):
        freq_list = []
        amplitude_list = []
        bin_max_frequency_value = math.ceil(len(frequency)/sliders_number)
        i = 0
        for i in range(0, sliders_number):
            freq_list.append(frequency[i*bin_max_frequency_value : (i+1)*bin_max_frequency_value])
            amplitude_list.append(amplitude[i*bin_max_frequency_value:(i+1)*bin_max_frequency_value])

        return freq_list, amplitude_list,bin_max_frequency_value

    def Sliders_generation( sliders_number,text):
            columns = st.columns(sliders_number)
            values = []
            for i in range(0, sliders_number):
                with columns[i]:
                    value = svs.vertical_slider( key= i, default_value=0, step=1, min_value=-20, max_value=20)
                    st.text(text[i])
                    if value == None:
                        value = 0.0
                    values.append(value)
                    
            return values   

        
    def convertToAudio(sr,signal):
        bytes_wav = bytes()
        byte_io = io.BytesIO(bytes_wav)
        write(byte_io,sr, signal.astype(np.float32))
        result_bytes = byte_io.read()
        return result_bytes

    def inverse(amp,phase):
        combined=np.multiply(amp,np.exp(1j*phase))
        inverse_combined=np.fft.irfft(combined)
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






    def plot_animation(df):
        brush = alt.selection_interval()
        chart1 = alt.Chart(df).mark_line().encode(
                x=alt.X('time', axis=alt.Axis(title='Time')),
                # y=alt.Y('amplitude', axis=alt.Axis(title='Amplitude')),
            ).properties(
                width=500,
                height=300
            ).add_selection(
                brush).interactive()
        
        figure = chart1.encode(
                    y=alt.Y('amplitude',axis=alt.Axis(title='Amplitude')))| chart1.encode(
                    y=alt.Y('amplitude after processing',axis=alt.Axis(title='Amplitude after'))).add_selection(
                brush)

        return figure

    def plotShow(data, idata,start_btn,pause_btn,resume_btn,value,sr):

        time1 = len(data)/(sr)
        if time1>1:
            time1 = int(time1)
        time1 = np.linspace(0,time1,len(data))   
        df = pd.DataFrame({'time': time1[::30], 
                            'amplitude': data[:: 30],
                            'amplitude after processing': idata[::30]}, columns=[
                            'time', 'amplitude','amplitude after processing'])
        N = df.shape[0]  # number of elements in the dataframe
        burst = 10      # number of elements (months) to add to the plot
        size = burst 
        
        step_df = df.iloc[0:st.session_state.size1]
        if st.session_state.size1 ==0:
            step_df = df.iloc[0:N]

        lines = Functions.plot_animation(step_df)
        line_plot = st.altair_chart(lines)
        line_plot= line_plot.altair_chart(lines)

        # lines = plot_animation(df)
        # line_plot = st.altair_chart(lines)
        N = df.shape[0]  # number of elements in the dataframe
        burst = 10      # number of elements (months) to add to the plot
        size = burst    #   size of the current dataset
        if start_btn:
            st.session_state.flag = 1
            for i in range(1, N):
                st.session_state.start=i
                step_df = df.iloc[0:size]
                lines = Functions.plot_animation(step_df)
                line_plot = line_plot.altair_chart(lines)
                size = i + burst 
                st.session_state.size1 = size
                time.sleep(.1)

        elif resume_btn: 
                st.session_state.flag = 1
                for i in range( st.session_state.start,N):
                    st.session_state.start =i 
                    step_df = df.iloc[0:size]
                    lines = Functions.plot_animation(step_df)
                    line_plot = line_plot.altair_chart(lines)
                    st.session_state.size1 = size
                    size = i + burst
                    time.sleep(.1)

        elif pause_btn:
                st.session_state.flag =0
                step_df = df.iloc[0:st.session_state.size1]
                lines = Functions.plot_animation(step_df)
                line_plot= line_plot.altair_chart(lines)



        if st.session_state.flag == 1:
            for i in range( st.session_state.start,N):
                    st.session_state.start =i 
                    step_df = df.iloc[0:size]
                    lines = Functions.plot_animation(step_df)
                    line_plot = line_plot.altair_chart(lines)
                    st.session_state.size1 = size
                    size = i + burst
                    time.sleep(.1)