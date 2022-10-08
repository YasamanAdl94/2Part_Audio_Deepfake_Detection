# 2Part_Audio_Deepfake_Detection

===============
This repository contains the implementation for two-part architecture using prosody features, background noise and semantic analysis for deepfake audio detection
![image.png] (https://github.com/YasamanAdl94/2Part_Audio_Deepfake_Detection/blob/ce936bcc855b3eee61c68d2790406d4bee28d1fd/image.png)


## Instructions

First, clone the repository locally, create and activate a conda environment, and install the requirements :

$ git clone https://github.com/YasamanAdl94/2Part_Audio_Deepfake_Detection.git
$ conda create --name RawGAT_ST_anti_spoofing python=3.8.8
$ conda activate RawGAT_ST_anti_spoofing
$ conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
$ pip install -r requirements.txt



## Experiments

### Dataset
ADD 2022 Challenge dataset has been used in this project.

### Training
To train the model run:

python main.py --loss=WCE   --lr=0.0001 --batch_size=5

### Testing

python main.py --loss=WCE --is_eval --eval --model_path='/path/to/your/best_model.pth' --eval_output='scores.txt'








