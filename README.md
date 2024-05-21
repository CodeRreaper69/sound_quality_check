# Audio Quality Comparison Tool

[Live App](https://sound-quality-analysis-by-dey-sourabh.streamlit.app/)  


## Overview

The Audio Quality Comparison Tool is a powerful application designed to compare the quality of two audio recordings: an original reference audio and a performance audio. It utilizes various audio analysis techniques to provide a comprehensive assessment of the performance audio in comparison to the original.

## Features

1. **Dynamic Time Warping (DTW) Distance Analysis**:
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
   - Allows visualization of how the frequency content changes over time in each recording.

8. **Audio Playback**:
   - Enables users to listen to the uploaded audio files directly within the app.

## How to Use

1. **Upload Audio Files**:
   - Upload the original reference audio file.
   - Upload the performance audio file.

2. **Start Analysis**:
   - Click on the appropriate buttons to start the numerical parameters analysis, graphical comparison, or real-time frequency analysis.

3. **View Results**:
   - The app will display various metrics and visualizations comparing the two audio files.

4. **Playback Audio**:
   - Listen to the uploaded audio files using the built-in audio playback feature.

## Benefits

- **Comprehensive Analysis**: Provides a detailed comparison using multiple audio features and metrics.
- **Visualization**: Offers clear visual representations of the differences between the two audio files.
- **Real-Time Feedback**: Allows real-time frequency distribution analysis to monitor changes dynamically.

## Limitations

- **Audio Quality**: The accuracy of the analysis can be affected by the quality of the uploaded audio files.
- **File Formats**: Currently supports common audio file formats such as WAV and MP3.
- **Performance**: Real-time analysis may be resource-intensive depending on the length and quality of the audio files.

## Future Scope

- **Additional Metrics**: Incorporate more audio features and metrics for an even more comprehensive analysis.
- **Enhanced Visualizations**: Improve the graphical representations and add interactive elements.
- **Support for More Formats**: Extend support to additional audio file formats.
- **Machine Learning Integration**: Use machine learning models to provide predictive analysis and automated feedback.

## Developer Information

Created by [Sourabh Dey](https://github.com/CodeRreaper69)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
