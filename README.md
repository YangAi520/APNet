# APNet: An All-Frame-Level Neural Vocoder Incorporating Direct Prediction of Amplitude and Phase Spectra
### Yang Ai, Zhen-Hua Ling

In our [paper](https://arxiv.org/xxx), 
we proposed APNet: An all-frame-level neural vocoder reconstructing speech waveforms from acoustic features by predicting amplitude and phase spectra directly.<br/>
We provide our implementation and pretrained models as open source in this repository.

**Abstract :**
This paper presents a novel neural vocoder named APNet which reconstructs speech waveforms from acoustic features by predicting amplitude and phase spectra directly. The APNet vocoder is composed of an amplitude spectrum predictor (ASP) and a phase spectrum predictor (PSP). The ASP is a residual convolution network which predicts frame-level log amplitude spectra from acoustic features. The PSP also adopts a residual convolution network using acoustic features as input, then passes the output of this network through two parallel linear convolution layers respectively, and finally integrates into a phase calculation formula to estimate frame-level phase spectra. Finally, the outputs of ASP and PSP are combined to reconstruct speech waveforms by inverse short-time Fourier transform (ISTFT). All operations of the ASP and PSP are performed at the frame level. We train the ASP and PSP jointly and define multi-level loss functions based on amplitude mean square error, phase anti-wrapping error, short-time spectral inconsistency error and time domain reconstruction error. Experimental results show that our proposed APNet vocoder achieves about 8x faster inference speed than HiFi-GAN v1 on a CPU due to the all-frame-level operations while its synthesized speech quality is comparable to HiFi-GAN v1. The synthesized speech quality of the APNet vocoder is also better than several equally efficient models. Ablation experiments also confirm that the proposed parallel phase estimation architecture is essential to phase modeling and the proposed loss functions are helpful for improving the synthesized speech quality.

Visit our [demo website](http://staff.ustc.edu.cn/~yangai/APNet/demo.html) for audio samples.

## Requirements
```
torch==1.8.1+cu111
numpy==1.21.6
librosa==0.9.1
tensorboard==2.8.0
soundfile==0.10.3
matplotlib==3.1.3
```

## Data Preparation
For training, write the list paths of training set and validation set to `input_training_wav_list` and `input_validation_wav_list` in `config.json`, respectively.

For inference, we provide two ways to read data:

(1) set `test_mel_load` to `0` in `config.json` and write the test set waveform path to `test_input_wavs_dir` in `config.json`, the inference process will first load the waveform, then extract the mel spectrogram, and finally synthesize the waveform through the vocoder;

(2) set `test_mel_load` to `1` in `config.json` and write the test set mel spectrogram (size is `80*frames`) path to `test_input_mels_dir` in `config.json`, the inference process will dierctly load the mel spectrogram, and then synthesize the waveform through the vocoder.

**Note :** The sampling rate of speech waveforms must be 16kHz in this version of the code.

## Training
Run using GPU:
```
CUDA_VISIBLE_DEVICES=0 python train.py
```
Using TensorBoard to monitor the training process:
```
tensorboard --logdir=cp_APNet/logs
```

## Inference:
Write the checkpoint path to `checkpoint_file_load` in `config.json`.

Run using GPU:
```
CUDA_VISIBLE_DEVICES=0 python inference.py
```
Run using CPU:
```
CUDA_VISIBLE_DEVICES=CPU python inference.py
```
