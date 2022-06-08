import time
import streamlit as st
from playsound import playsound

class Sound():

    def __init__(self):
        audio_file = open('asset/alarm.wav', 'rb')
        self.audio_bytes = audio_file.read()

    def alarm(self):
        st.audio(self.audio_bytes, format="audio/wav")

def play_alarm():
    playsound('asset/alarm.wav', False)


class TimeForSoundChecker:

    def __init__(self):
        self.time_last_called = time.time() * 1000

    def has_been_a_second(self):
        current = time.time() * 1000
        if (current - self.time_last_called) > 1000:
            self.time_last_called = current
            return True
        return False

def has_violations(predictions):
    if len(predictions) == 0: return False
    for prediction in predictions:
        if prediction[0] in [1, 3]:
            return True
    return False