import matplotlib.pyplot as plt
import numpy as np
from scipy import fft
import scipy.io.wavfile as wavfile
import scipy.fftpack as fftpk
from statistics import mean 


def fft_filt(signal):
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

waveout = '/home/mher/Project_Python/v4/denoise.wav'
s_rate,signal = wavfile.read("/home/mher/Project_Python/v4/with_noise.wav") 
#sample_freq,signal_array=wavfile.read("/home/mher/Project_Python/v4/with_noise.wav")
t_audio=len(signal)/float(s_rate)
times=np.linspace (0,t_audio,num=len(signal))
print("recording duration is :",t_audio)


FFT = fft.fft(signal)
freqs = fftpk.fftfreq(len(FFT), (1.0/s_rate))

# :)
import warnings
warnings.filterwarnings('ignore')

plt.figure(figsize=(12, 5))
plt.subplot(211)
plt.plot(freqs[range(len(FFT)//2)], FFT[range(len(FFT)//2)])                                                          
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude before filter')

plt.subplot(212)
FFT_filt=fft_filt(signal)
plt.plot(freqs[range(len(FFT_filt)//2)], FFT_filt[range(len(FFT_filt)//2)])   
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude after filter')
plt.show()

plt.figure(figsize=(24, 10))
plt.subplot(211)
plt.plot(times,signal)
plt.xlabel("Time (seconds) -->")
plt.ylabel("Amplitude")

signal=(np.real(np.fft.ifft(FFT_filt)))/50000
plt.subplot(212)
plt.plot(times,signal*50000)
plt.xlabel("Time (seconds) -->")
plt.ylabel("Amplitude of result")
plt.show()

wavfile.write(waveout, s_rate, signal)