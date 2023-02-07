import wave
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile
import scipy
import scipy.fftpack as fftpk
import numpy as np
import plotly.graph_objects as go
import IPython
from statistics import mean 
"""
def filter_signal(th):
    f_s = fft_filter(th)
    return np.real(np.fft.ifft(f_s))
def fft_filter(perc):
    fft_signal = np.fft.fft(signal)
    fft_abs = np.abs(fft_signal)
    th=perc*(2*fft_abs[0:int(len(signal)/2.)]/((len(signal)/2.))).max()
    fft_tof=fft_signal.copy()
    fft_tof_abs=np.abs(fft_tof)
    fft_tof_abs=2*fft_tof_abs/((len(signal)/2.))
    fft_tof[fft_tof_abs<=th]=0
    return fft_tof
def fft_filter_amp(th):
    fft = np.fft.fft(signal)
    fft_tof=fft.copy()
    fft_tof_abs=np.abs(fft_tof)
    fft_tof_abs=2*fft_tof_abs/((len(signal)/2.))
    fft_tof_abs[fft_tof_abs<=th]=0
    return fft_tof_abs[0:int(len(fft_tof_abs)/2.)]
"""
def fft_filt(num1,num2,signal):
    fft=np.fft.fft(signal)
    mean_value = mean(abs(fft))
    threshold  = 1.1*mean_value
    fft[abs(fft) < threshold] = 0
    """ Alternative filter
    for i in range(len(fft)):
        if abs(fft[i])>=num1 and abs(fft[i])<=num2:
            fft[i]=0  
    """     
    return fft


audio_path=wave.open("/home/mher/Project_Python/v4/with_noise.wav","rb")
waveout = '/home/mher/Project_Python/v4/denoise.wav'
reduced_noise='/home/mher/Project_Python/v4/reduced_noise.wav'
sr=print("sr:",audio_path.getframerate())
sample_freq=audio_path.getframerate()   #sr
n_samples=audio_path.getnframes()
signal_wave=audio_path.readframes(-1)
t_audio=n_samples/float(sample_freq)
print(t_audio)
signal_array=np.frombuffer(signal_wave,dtype=np.int16)
times=np.linspace (0,t_audio,num=n_samples)

s_rate,signal = wavfile.read("/home/mher/Project_Python/v4/with_noise.wav") 
s_reduced,signal_reduced=wavfile.read("/home/mher/Project_Python/v4/reduced_noise.wav")

FFT = scipy.fft.fft(signal)
freqs = fftpk.fftfreq(len(FFT), (1.0/s_rate))

"""
plt.figure(figsize=(12, 5))
plt.subplot(211)
plt.plot(freqs[range(len(FFT)//2)], FFT[range(len(FFT)//2)])                                                          
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
"""

print(len(FFT)//2)
plt.figure(figsize=(12, 5))
plt.subplot(211)
FFT_filt=fft_filt(50000,500000,signal)
plt.plot(freqs[range(len(FFT_filt)//2)], FFT_filt[range(len(FFT_filt)//2)])   
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')


# FFT FIGURE OF REDUCED NOISE
plt.subplot(212)
FFT_reduced = scipy.fft.fft(signal_reduced)
freqs_reduced = fftpk.fftfreq(len(FFT_reduced), (1.0/s_reduced))
plt.plot(freqs_reduced[range(len(FFT_reduced)//2)], FFT_reduced[range(len(FFT)//2)])   
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude of well reducec noise')
plt.show()

plt.figure(figsize=(24, 10))
plt.subplot(211)
plt.plot(times,signal_array)
plt.xlabel("Time (seconds) -->")
plt.ylabel("Amplitude")

signal=(np.real(np.fft.ifft(FFT_filt)))/50000
plt.subplot(212)
plt.plot(times,signal*50000)
plt.xlabel("Time (seconds) -->")
plt.ylabel("Amplitude of result")

plt.show()
wavfile.write(waveout, s_rate, signal)



