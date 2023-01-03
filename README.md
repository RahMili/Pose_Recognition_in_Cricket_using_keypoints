Official Repository for the paper "Pose Recognition in Cricket using Keypoints", paper id: 218, presented at IEEE UPCON 2022.
## Description
A method for pose recognition of umpire in the game of cricket. This can in future be used for generating cricket highlights leveraging the efficiency of pose estimation models and Artificial Neural Networks.

Our work can identify 4 actions (Six, No-ball, Out, Wide) and No action (any other pose besides the 4 will be categorized into No action).
## Setup
1. Create Python env>= 3.7
2. Install Python packages specified in requirements.txt in the Python env
3. Create 'data' folder and add 'train' and 'test' folders (check config.ini in config folder)
4. Get the training data into 'train' folder and testing data in 'test' folder.
5. Create 'models' folder
6. Download the pretrained model and put it inside 'models' folder.
7. Pretrained model download link 'https://drive.google.com/file/d/1XeiT111eXo0yjjkuKowBrgGlkhQcuoHw/view?usp=sharing'
## Usage
Set PYTHONPATH = '..../Pose-Recognition-in-Cricket-using-keypoints'

To train the model - run train.py (Not required as we have pretrained model, but can try if you really need to)

To test on pretrained model - run inference.py