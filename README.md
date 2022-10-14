# 2Part_Audio_Deepfake_Detection Initial Prototype

This repository contains the implementation for two-part architecture using prosody features, background noise and semantic analysis for deepfake audio detection
![image](https://user-images.githubusercontent.com/61777099/194679033-9c61bc9f-9bc6-415e-be1e-579ac4109d8c.png)


## Instructions

First, clone the repository locally, create and activate a conda environment, and install the requirements :

$ git clone https://github.com/YasamanAdl94/2Part_Audio_Deepfake_Detection.git

$ conda create --name RawGAT_ST_anti_spoofing python=3.8.8

$ conda activate RawGAT_ST_anti_spoofing

$ conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia

$ pip install -r requirements.txt

## Libraries used and their licenses

    Librosa: ISC License
    Soundfile: BSD License (BSD 3-Clause License)
    DisVoice: MIT License
    my-voice-analysis: MIT License
## Feature_Extraction

### Background noise extraction

Using Librosa library, we can extract and separate background and foreground information of speech samples. 

### Prosody Features

Prosody features are extracted from audio files based on duration, fundamental frequency and energy. 
103 features include:

      V: voiced
      U: Unvoiced
      1-6 F0-contour: Avg., Std., Max., Min., Skewness, Kurtosis
      7-12 Tilt of a linear estimation of F0 for each voiced segment: Avg., Std., Max., Min., Skewness, Kurtosis
      13-18 MSE of a linear estimation of F0 for each voiced segment: Avg., Std., Max., Min., Skewness, Kurtosis
      19-24 F0 on the first voiced segment: Avg., Std., Max., Min., Skewness, Kurtosis
      25-30 F0 on the last voiced segment: Avg., Std., Max., Min., Skewness, Kurtosis
      31-34 energy-contour for voiced segments: Avg., Std., Skewness, Kurtosis
      35-38 Tilt of a linear estimation of energy contour for V segments: Avg., Std., Skewness, Kurtosis
      39-42 MSE of a linear estimation of energy contour for V segment: Avg., Std., Skewness, Kurtosis
      43-48 energy on the first voiced segment: Avg., Std., Max., Min., Skewness, Kurtosis
      49-54 energy on the last voiced segment: Avg., Std., Max., Min., Skewness, Kurtosis
      55-58 energy-contour for unvoiced segments: Avg., Std., Skewness, Kurtosis
      59-62 Tilt of a linear estimation of energy contour for U segments: Avg., Std., Skewness, Kurtosis
      63-66 MSE of a linear estimation of energy contour for U segments: Avg., Std., Skewness, Kurtosis
      67-72 energy on the first unvoiced segment: Avg., Std., Max., Min., Skewness, Kurtosis
      73-78 energy on the last unvoiced segment: Avg., Std., Max., Min., Skewness, Kurtosis
      79 Voiced rate: Number of voiced segments per second
      80-85 Duration of Voiced: Avg., Std., Max., Min., Skewness, Kurtosis
      86-91 Duration of Unvoiced: Avg., Std., Max., Min., Skewness, Kurtosis
      92-97 Duration of Pauses: Avg., Std., Max., Min., Skewness, Kurtosis
      98-103 Duration ratios: Pause/(Voiced+Unvoiced), Pause/Unvoiced, Unvoiced/(Voiced+Unvoiced),Voiced/(Voiced+Unvoiced), Voiced/Puase, Unvoiced/Pause


### Phonation Features
 

There are seven phonation features which are extracted:

    First derivative of the fundamental Frequency
    Second derivative of the fundamental Frequency
    Jitter
    Shimmer
    Amplitude perturbation quotient
    Pitch perturbation quotient
    Logaritmic Energy

### Semantic Analysis

A python library "my-voice-analysis" is used to analyze voices by breaking utterances and detecting fundamental frequency contours. The features that can be extracted using this library are:
    
    number_ of_syllables     
    number_of_pauses         
    rate_of_speech             
    articulation_rate          
    speaking_duration       
    original_duration       
    balance                 
    f0_mean               
    f0_std                
    f0_median             
    f0_min                   
    f0_max                   
    f0_quantile25            
    f0_quan75               

### Dataset
ADD 2022 Challenge dataset has been used in this project.

### Training
To train the model run:

python main.py --loss=WCE   --lr=0.0001 --batch_size=5

(Other values such as epoch numbers are set as default but can be changed if given in the command line e.g. --num_epochs=500)

Note: The dataset path should be changed to the corresponding folder. Data processing will result in a pickle file with label and each feature. Pytorch dataloader will load this file to forward to train set.

### Testing

python evaluation.py --loss=WCE --model_path='/path/to/your/best_model.pth' --eval_output='scores.txt'

Track 1: **Fully fake** audios generated using the TTS and VC algorithms with various background noises

Track 2: **Partially fake** speech generated by manipulating the original real utterances with real or synthesized audios

The obtained EER values so far are as below:

| Features                    | Track 1       |          Track 2          |
| --------------------------- |:-------------:| -------------------------:|
| Background noise Analysis   | In progress       |          In progress                 |
| Prosody variations          |     In progress  |           47.05%         |
| Phonation (samples in ADD too short)                   |        78%    |            68%            |
| Semantic features           |   In progress |           In progress     |


Current results indicate a possible future for prosodic features, phonation and background noise analysis for deepfake audio detection. Structural features for audio are being used more often to detect spoofed audio. Graph Attention Networks for anti spoofing were first proposed in [https://arxiv.org/abs/2104.03654]. We will continue to explore structural features in two part architecture to better understand the underlying characteristics of audios so we can better distinguish fake samples. Specially for track 2 of ADD dataset which has fake segments inside real audio, structural features can analyze formants (parts of speech with energy peak) or statistical aspects of speech. There is more to structural features which we continue to study.

