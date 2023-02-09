import numpy as np
import matplotlib.pyplot as plt
import nlms
from scipy.io import wavfile

wavebefore='/home/mher/Project_Python/echo-cancel/before.wav'
waveout = '/home/mher/Project_Python/echo-cancel/after.wav' # Defining the output wave file

step = 0.05 # Step size
M = 50 # Number of filter taps in adaptive filter


sfs, u = wavfile.read('/home/mher/Project_Python/echo-cancel/Sounds/Sender.wav')
lfs, v = wavfile.read('/home/mher/Project_Python/echo-cancel/Sounds/Listener.wav')

u = np.frombuffer(u, np.int16)
u = np.float64(u)

v = np.frombuffer(v, np.int16)
v = np.float64(v)

# Generate the fedback signal d(n) by a) convolving the sender's voice with randomly chosen coefficients assumed to emulate the listener's room 
# characteristic, and b) mixing the result with listener's voice, so that the sender hears a mix of noise and echo in the reply.

coeffs = np.concatenate(([0.8], np.zeros(8), [-0.7], np.zeros(9), [0.5], np.zeros(11), [-0.3], np.zeros(3),[0.1], np.zeros(20), [-0.05]))
d = np.convolve(u, coeffs)
d = d/20.0
v = v/20.0
d = d[:len(v)] 
d = d + v - (d*v)/256.0   
d = np.round(d,0)

# Hear how the mixed signal sounds before proceeding with the filtering.
dsound = d.astype('int16')
wavfile.write(wavebefore, lfs, dsound)


# Apply adaptive filter
y, e, w = nlms.nlms(u[:len(d)], d, M, step, returnCoeffs=True)

e = e.astype('int16')
wavfile.write(waveout, lfs, e)