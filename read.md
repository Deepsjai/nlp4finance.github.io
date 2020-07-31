
# Introduction
Audio files are notoriously large when uncompressed. For example, CD quality audio, sampled at 44 kHz, has a bitrate of 1,411 kbits per second (kbps). On a 700 MB CD, there is only enough capacity for about an hour of music.

Consequently audio compression is essential to reducing file sizes to more practical levels, enabling efficient storage and transmission. MP3 is the most common lossy compression algorithm which uses spectral transforms to harness the sparisity and the perceptual limitation of human hearing.

The MP3 codec typically compresses to a bitrate between 128 kbit/s and 320 kbit/s. This demo compares audio bitrates of a sample song from 1kbit/s to 320 kbit/s.

Traditional methods of music compression typcially use deterministic algorithms, which rely on identifying features and patterns in the frequency domain. As deep networks excel at capturing complex patterns and extracting nonlinear features, we analyze their ability to condense a song's patterns into a compressed lower-dimensional space.

Once our compression and extraction framework is in place, we can leverage this compressed space or 'latent space'. The latent space is a more compact version of the audio sample, and is therefore useful when attempting to perform classification. We build another network based on the latent space as opposed to the original input space to perform genre classification. Previous works [3] [4] have used similar methods on the task of genre prediction, using a spectral representation of audio with CNNs and RNNs.

# Dataset
We use the FMA dataset [5]. This is an open dataset of ~1 TB of songs from many artists and genres. For this project we use the small version of the dataset containing 8000 songs from 8 genre categories. We used a 70-30 split between train and test set. The choice to use small version was due to unavailability of computing resources needed for larger versions of the dataset.

Unsupervised Audio Compression
A deep autoencoder is a special type of feedforward neural network which can be used in denoising and compression [2]. In this architecture, the network consists of an encoder and decoder module. The encoder learns to compress a high-dimensional input X to a low-dimensional latent space z. This "bottleneck" forces the newtork to learn a compression to reduce the information in X to size of Z. The decoder then attempts to faithfully reconstruct the output with minimal error. Both the encoder and decoder are implemented as convolutional neural networks.

Clearly, it is impossible to reconstruct the input with zero error, so the network learns a lossy compression. The network can discover patterns in the input to reduce the data dimensionality required to fit through the bottleneck. The network is penalized with an L2 reconstruction loss. This is a completely unsupervised method of training that provides very rich supervision.

Autoencoder

# Autoencoder network structure. Image credit to Lilian Weng

Frequency-Domain Autoencoder
There are several choices of input space which are critical to achieving good performance. In keeping with other similar approaches [1], we convert the audio signal into a spectrogram using a short-time-fourier-transform (STFT). This converts the song into an "image", with time on one axis and frequency on another. This has advantages in that it is more human-interpretable, and a broad family of techniques from computer vision can be used, as this is thought of as a 2D image.

# Spectrogram

# Model details
freq_ae_model

# Loss function
An RMSE reconstruction loss is used to train the model. This model effectively penalizes large errors, with less weight given to small deviations. As seen in the next section, this directly optimizes for our evaluation metric.

# Compression Evaluation Metric
Music is fundamentally subjective. Thus generating a quantitative evaluation metric for our compression algorithm is very difficult. It is not possible to naively compare the reconstructed time domain signals, as completely different signals can sound the same. For example, phase shift, or small uniform frequency shifts are imperceptible to the human ear. A naive loss in the time domain would heavily penalise this.

# Phase Shift

On the other hand, a time domain loss does not adequately capture high frequencies and low volumes. As human perception of sound is logarithmic, and low frequencies typically have higher amplitude, a time domain loss under-weights high frequencies and results in a muffled, underwater-sounding output.

We follow the approach of [1] and instead use an RMSE metric by directly comparing the frequency spectra across time. This has the benefit of considering low amplitudes and high frequencies, and is perceptually much closer.

# RMSE Loss

Time-Domain Autoencoder
Our main motivation for this approach is to build an end-to-end network so that it can potentially learn a more compressed representation. This approach is inspired from computer vision where people moved from a classical pipeline of feature design to end-to-end deep models.

Learning on a time domain signal saves space too as the spectral domain of an audio signal is sparse. We can directly go to a more efficient representation right after the first layer.

Model Details
time_domain_autoencoder

Loss functions
Even though an RMSE loss in the time domain is not the best choice from a point of view of audio perception, we found that it worked better than loss computation in spectral or log-spectral domain.

