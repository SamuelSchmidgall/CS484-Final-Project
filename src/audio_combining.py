import os
import subprocess


from pydub import AudioSegment


for i in os.listdir(r"C:\Users\Samuel\Desktop\selective_hearing\wav_data"):
    for j in os.listdir(r'C:\Users\Samuel\Desktop\selective_hearing\sine_noise'):
        try:
            noise_fl = r'C:\Users\Samuel\Desktop\selective_hearing\sine_noise/{}'.format(j)
            audio_fl = r"C:\Users\Samuel\Desktop\selective_hearing\wav_data/{}".format(i)
            sound1 = AudioSegment.from_file(noise_fl)
            sound2 = AudioSegment.from_file(audio_fl)
            combined = sound1.overlay(sound2)
            combined.export(r"C:\Users\Samuel\Desktop\selective_hearing\combined_noise_audio/{}&{}".format(j, i), format='wav')
        except Exception as e:
            print(e)
