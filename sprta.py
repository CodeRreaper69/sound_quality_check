import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import streamlit as st
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from scipy.ndimage import gaussian_filter1d
import time
import os
import ffmpeg

st.set_page_config(page_title="AUDIO-ANALYSIS", page_icon="👂")

# Streamlit app
st.title(':violet[Audio Quality Comparison]')

# Information corner
st.markdown("""
---
Created by [Sourabh Dey](https://github.com/CodeRreaper69)
""")

with st.expander(":orange[What is this and how it works?]"):
    st.markdown("""
    ## Audio Quality Analysis Tool

    This tool allows you to compare the quality of two audio recordings: an original reference audio and your performance audio. The comparison is done using several audio features and metrics to give you a comprehensive analysis of how well your performance matches the original.

    ### How It Works

    1. **Dynamic Time Warping (DTW) Distance**:
        - Calculates the similarity between the original and performance audio using Mel-Frequency Cepstral Coefficients (MFCC) and pitch features.
        - Lower DTW distance indicates higher similarity.

    2. **Tempo Analysis**:
        - Measures the beats per minute (BPM) of both audio files.
        - Shows the percentage deviation in tempo.

    3. **Zero Crossing Rate (ZCR) Analysis**:
        - Counts the rate at which the audio signal changes sign.
        - Higher ZCR can indicate higher levels of noise or fricative sounds.

    4. **Energy Analysis**:
        - Computes the Root Mean Square (RMS) energy of the audio signals.
        - Indicates the loudness and energy of the recordings.

    5. **Harmonic-Percussive Source Separation (HPSS)**:
        - Separates the audio into harmonic and percussive components.
        - Analyzes the energy of both components to compare the tonal and rhythmic elements of the recordings.

    6. **Spectral Contrast**:
        - Measures the difference in amplitude between peaks and valleys in a sound spectrum.
        - Helps in understanding the timbral texture of the audio.

    7. **Real-Time Frequency Distribution**:
        - Displays a real-time comparison of the frequency distribution of both the original and performance audio.
        - Allows you to visualize how the frequency content changes over time in each recording.
        - This is particularly useful for identifying differences in specific frequency bands, such as bass, midrange, and treble.

    ### What This Shows

    - **DTW Distance**: Gives you an idea of how closely your performance matches the original in terms of pitch and timbre.
    - **Tempo**: Shows if your performance is faster or slower than the original.
    - **ZCR**: Indicates the noisiness or clarity of your performance.
    - **Energy Levels**: Compares the loudness and dynamic range.
    - **HPSS**: Provides insight into the harmonic and percussive quality of your performance.
    - **Spectral Contrast**: Visualizes the timbral differences between the original and performance audio.
    - **Real-Time Frequency Distribution**: Provides a dynamic view of the frequency content, showing how the spectrum of your performance matches or deviates from the original over time.

    ### Supported Audio Types

    - The tool accepts all common audio file formats such as mp3, wav, flv, ogg, m4a, flac, aiff, aac, wma, webm.
    - Ensure that the audio files are of good quality and not corrupted for accurate analysis.

    This analysis is presented through various graphs and numerical comparisons, helping you to visually and quantitatively assess your performance.
    """)

TMP_DIR = "temp_files"
if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)

# Supported file types for upload and conversion
support_upload_list = ["mp3", "wav", "flv", "ogg", "m4a", "flac", "aiff", "aac", "wma", "webm"]
# File upload

audio_file1 = st.file_uploader(":blue[Upload Your Original Audio File ]", type=support_upload_list)
audio_file2 = st.file_uploader(":blue[Upload Your Performance Audio File ]", type=support_upload_list)

# Function to convert any audio file to mp3
def convert_to_mp3(uploaded_file):
    file_name = uploaded_file.name
    current_file_ext = file_name.split(".")[-1]
    AUDIO_FILE = os.path.join(TMP_DIR, file_name)

    # Save the uploaded file to the temporary directory
    with open(AUDIO_FILE, "wb") as f:
        f.write(uploaded_file.read())

    # Convert to MP3 automatically
    output_file = AUDIO_FILE.replace(current_file_ext, 'mp3')
    try:
        ffmpeg.input(AUDIO_FILE).output(output_file).run(overwrite_output=True)
        return output_file
    except ffmpeg.Error as e:
        st.error(f"An error occurred: {e}")
        return AUDIO_FILE  # Fallback to original file if conversion fails

def cleanup_temp_files():
    for file_name in os.listdir(TMP_DIR):
        file_path = os.path.join(TMP_DIR, file_name)
        try:
            os.remove(file_path)
        except Exception as e:
            st.error(f"Error deleting file {file_path}: {e}")







if audio_file1 and audio_file2:
    # Convert uploaded files to MP3
    st.spinner("Converting files to acceptable format...")
    uploaded_file1 = convert_to_mp3(audio_file1)
    if uploaded_file1:
        st.audio(uploaded_file1, format='audio/wav')
    uploaded_file2 = convert_to_mp3(audio_file2)
    if uploaded_file2:
        st.audio(uploaded_file2, format='audio/wav')
    
    
  
cleanup_button = st.button("```Kindly Click this button to clean up Temporary files before you leave to save space```")
if cleanup_button:
   cleanup_temp_files()    


# Function to compute and plot spectrogram
def plot_spectrogram(y, sr, ax, title):
    S = librosa.stft(y)
    S_dB = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='log', ax=ax)
    ax.set_title(title)
    return img

# Function to extract pitch features using Librosa
def extract_pitch(y, sr):
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    avg_pitch = librosa.feature.rms(S=magnitudes * pitches)
    return avg_pitch


# Function to load and preprocess audio
def load_audio(file):
    y, sr = librosa.load(file, sr=None)
    return y, sr

# Function to compute and plot spectrogram
def plot_spectrogram(y, sr, ax, title):
    S = librosa.stft(y)
    S_dB = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='log', ax=ax)
    ax.set_title(title)
    return img


tab1, tab2, tab3 = st.tabs(["Numerical Parameters Analysis","Graphical Analysis","Real Time Frequency Analysis"])


with tab1:
    num_button = st.button(":red[__Click to start the process if files are uploaded successfully__]")
    if audio_file1 is not None and audio_file2 is not None and num_button is True:
        y1, sr1 = librosa.load(uploaded_file1, sr=None)
        y2, sr2 = librosa.load(uploaded_file2, sr=None)

        # Lambda functions for percentage calculation
        x = lambda a, b: abs(((a - b) / a) * 100)
        v = lambda a, b: abs(((a - b) / a) * 100)

        # Extract MFCC features
        mfcc_original = librosa.feature.mfcc(y=y1, sr=sr1)
        mfcc_performance = librosa.feature.mfcc(y=y2, sr=sr2)

        # Extract pitch features
        pitch_original = extract_pitch(y=y1, sr=sr1)
        pitch_performance = extract_pitch(y=y2, sr=sr2)

        st.subheader("Dynamic Time Warp Distance Analysis")
        distance_mfcc, _ = fastdtw(mfcc_original.T, mfcc_performance.T, dist=euclidean)
        st.write("DTW Distance (MFCC):", distance_mfcc)
        distance_pitch, _ = fastdtw(pitch_original.T, pitch_performance.T, dist=euclidean)
        st.write("DTW Distance (Pitch):", distance_pitch)

        st.subheader("Tempo Analysis")
        tempo_original = librosa.beat.tempo(y=y1, sr=sr1)[0]
        tempo_performance = librosa.beat.tempo(y=y2, sr=sr2)[0]
        st.write("Tempo (Original):", tempo_original)
        st.write("Tempo (Performance):", tempo_performance)
        v_tempo = v(tempo_original, tempo_performance)
        st.write(f"Percentage Deviation: {v_tempo:.2f}%")

        st.subheader("Zero Crossing Rate Analysis")
        zcr_original = librosa.feature.zero_crossing_rate(y=y1)
        zcr_performance = librosa.feature.zero_crossing_rate(y=y2)
        avg_zcr_original = np.mean(zcr_original)
        avg_zcr_performance = np.mean(zcr_performance)
        st.write("Average ZCR (Original):", avg_zcr_original)
        st.write("Average ZCR (Performance):", avg_zcr_performance)
        v_zcr = v(avg_zcr_original, avg_zcr_performance)
        st.write(f"Percentage Deviation: {v_zcr:.2f}%")

        st.subheader("Energy Analysis")
        energy_original = librosa.feature.rms(y=y1)
        energy_performance = librosa.feature.rms(y=y2)
        mean_energy_original = np.mean(energy_original)
        mean_energy_performance = np.mean(energy_performance)
        st.write("Mean Energy (Original):", mean_energy_original)
        st.write("Mean Energy (Performance):", mean_energy_performance)
        v_mean_energy = v(mean_energy_original, mean_energy_performance)
        st.write(f"Percentage Deviation: {v_mean_energy:.2f}%")

        st.subheader("Harmonic-Percussive Source Separation")
        y_harmonic, y_percussive = librosa.effects.hpss(y=y1)
        y_harmonic_performance, y_percussive_performance = librosa.effects.hpss(y=y2)
        energy_harmonic = np.mean(librosa.feature.rms(y=y_harmonic))
        energy_percussive = np.mean(librosa.feature.rms(y=y_percussive))
        energy_harmonic_performance = np.mean(librosa.feature.rms(y=y_harmonic_performance))
        energy_percussive_performance = np.mean(librosa.feature.rms(y=y_percussive_performance))
        st.write("Energy Harmonic (Original):", energy_harmonic)
        st.write("Energy Harmonic (Performance):", energy_harmonic_performance)
        v_energy_harmonic = v(energy_harmonic, energy_harmonic_performance)
        st.write(f"Percentage Deviation: {v_energy_harmonic:.2f}%")
        st.write("Energy Percussive (Original):", energy_percussive)
        st.write("Energy Percussive (Performance):", energy_percussive_performance)
        v_energy_percussive = v(energy_percussive, energy_percussive_performance)
        st.write(f"Percentage Deviation: {v_energy_percussive:.2f}%")

        # Overall audio quality
        deviation = np.mean([v_tempo, v_zcr, v_mean_energy, v_energy_harmonic, v_energy_percussive])
        st.write("----------------------------------------------------------------")
        st.write(f"**Overall Audio Quality Compared to the Original: {100 - deviation:.2f}%**")
        st.write("----------------------------------------------------------------")
        #st.caption("IGNORE THE ERROR MESSAGE IF IT IS VISIBLE, SINCE IT IS IN A TESTING PHASE")


        


#real_time_button = st.button(':orange[Start Real-Time Comparison]')
with tab2:
    grp_button = st.button(":green[Click to generate the comparison graph]")
    if audio_file1 is not None and audio_file2 is not None and grp_button is True:
        y1, sr1 = librosa.load(uploaded_file1, sr=None)
        y2, sr2 = librosa.load(uploaded_file2, sr=None)
        st.subheader("Graphical Comparison")
        # Extract MFCC features
        mfcc_original = librosa.feature.mfcc(y=y1, sr=sr1)
        mfcc_performance = librosa.feature.mfcc(y=y2, sr=sr2)

        # Extract pitch features
        pitch_original = extract_pitch(y=y1, sr=sr1)
        pitch_performance = extract_pitch(y=y2, sr=sr2)
        # Plot comparison
        fig, ax = plt.subplots(3, 2, figsize=(14, 12))

        librosa.display.waveshow(y1, sr=sr1, ax=ax[0, 0])
        ax[0, 0].set(title='Original Audio Waveform', xlabel='Time (seconds)', ylabel='Amplitude')

        librosa.display.waveshow(y2, sr=sr2, ax=ax[0, 1])
        ax[0, 1].set(title='Performance Audio Waveform', xlabel='Time (seconds)', ylabel='Amplitude')

        spectral_contrast_original = librosa.feature.spectral_contrast(y=y1, sr=sr1)
        spectral_contrast_performance = librosa.feature.spectral_contrast(y=y2, sr=sr2)
        
        img = librosa.display.specshow(spectral_contrast_original, sr=sr1, x_axis='time', ax=ax[1, 0])
        fig.colorbar(img, ax=ax[1, 0], format='%+2.0f dB')
        ax[1, 0].set(title='Spectral Contrast (Original)', xlabel='Time (seconds)', ylabel='Frequency Bands')

        img = librosa.display.specshow(spectral_contrast_performance, sr=sr2, x_axis='time', ax=ax[1, 1])
        fig.colorbar(img, ax=ax[1, 1], format='%+2.0f dB')
        ax[1, 1].set(title='Spectral Contrast (Performance)', xlabel='Time (seconds)', ylabel='Frequency Bands')

        img = librosa.display.specshow(mfcc_original, sr=sr1, x_axis='time', ax=ax[2, 0])
        fig.colorbar(img, ax=ax[2, 0])
        ax[2, 0].set(title='MFCC (Original)', xlabel='Time (seconds)', ylabel='MFCC Coefficients')

        img = librosa.display.specshow(mfcc_performance, sr=sr2, x_axis='time', ax=ax[2, 1])
        fig.colorbar(img, ax=ax[2, 1])
        ax[2, 1].set(title='MFCC (Performance)', xlabel='Time (seconds)', ylabel='MFCC Coefficients')

        plt.tight_layout()
        st.pyplot(fig)
        #st.caption("IGNORE THE ERROR MESSAGE IF IT IS VISIBLE, SINCE IT IS IN A TESTING PHASE")




with tab3:
    a = st.button(":blue[Start/Refresh]")
    #st.caption("IGNORE THE ERROR MESSAGE IF IT IS VISIBLE, SINCE IT IS IN A TESTING PHASE")

    if audio_file1 and audio_file2:
        try:
            y1, sr1 = load_audio(uploaded_file1)
            y2, sr2 = load_audio(uploaded_file2)
        except Exception as e:
            st.error("Try refreshing or going to next Tab")
            st.stop()  # Stop execution if there is an error

        st.subheader('Real-Time Frequency Comparison')

        # Create placeholders for the plots
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))
        img1 = plot_spectrogram(y1[:sr1 * 5], sr1, ax[0], 'Original Audio')
        img2 = plot_spectrogram(y2[:sr2 * 5], sr2, ax[1], 'Performance Audio')

        placeholder = st.pyplot(fig)

        analyzing = False

        if st.button('Start Real-Time Comparison'):
            analyzing = True

        if st.button('Stop Real-Time Comparison'):
            analyzing = False

        # Real-time comparison loop
        hop_length = sr1 // 2  # Half-second hop length for smoother transitions
        window_length = sr1 * 5  # 5-second window length

        while analyzing:
            for start in range(0, min(len(y1), len(y2)) - window_length, hop_length):
                if not analyzing:
                    break

                y1_segment = y1[start:start + window_length]
                y2_segment = y2[start:start + window_length]

                # Clear the axes
                ax[0].cla()
                ax[1].cla()

                # Plot the updated spectrograms
                plot_spectrogram(y1_segment, sr1, ax[0], 'Original Audio')
                plot_spectrogram(y2_segment, sr2, ax[1], 'Performance Audio')

                # Update the plot
                placeholder.pyplot(fig)

                # Wait for a short interval to simulate real-time update
                time.sleep(0.5)
    
    elif audio_file1 is None or audio_file2 is None:
        st.toast(":red[**Upload Audio Files**]")
    else:
        st.toast(":red[Click on the Button]")




#elif uploaded_file1 is None and uploaded_file2 is None:
   # st.toast(":green[Upload audio files to continue]")
