import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

CHUNK = 256
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 11250

audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

fig, (ax_wave, ax_freq) = plt.subplots(2, 1)
x_time = np.arange(0, CHUNK)
x_freq = np.fft.rfftfreq(CHUNK, d=1/RATE)
line_wave, = ax_wave.plot(x_time, np.zeros(CHUNK), color='green')
line_freq, = ax_freq.plot(x_freq, np.zeros(len(x_freq)), color='purple')

ax_wave.set_title("Time Domain (Waveform)")
ax_wave.set_xlabel("Samples")
ax_wave.set_ylabel("Amplitude")
ax_wave.set_xlim(0, CHUNK)
ax_wave.set_ylim(-32768, 32767)

ax_freq.set_title("Frequency Domain (Spectrum)")
ax_freq.set_xlabel("Frequency (Hz)")
ax_freq.set_ylabel("Magnitude")
ax_freq.set_xlim(0, RATE / 2)
ax_freq.set_ylim(0, 10000)

def update(frame):
    data = stream.read(CHUNK, exception_on_overflow=False)
    audio_data = np.frombuffer(data, dtype=np.int16)
    line_wave.set_ydata(audio_data)
    fft_data = np.abs(np.fft.rfft(audio_data))
    line_freq.set_ydata(fft_data)
    ax_freq.set_ylim(0, np.max(fft_data) * 1.1)  # Dynamically update y-axis here
    return line_wave, line_freq


ani = FuncAnimation(fig, update, interval=50)

plt.tight_layout()
plt.show()

stream.stop_stream()
stream.close()
audio.terminate()
#ax_freq.set_ylim(0, np.max(fft_data) * 1.1)

