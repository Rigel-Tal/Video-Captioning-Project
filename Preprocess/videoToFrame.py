import cv2
import os

import numpy as np

# torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

import torchvision.models as models

# PILLOW
from PIL import Image

nframe=16
curr_path = "C:/Users/VISHWANATHAN VIVEK S/Desktop/Projects/VideoCaptioning"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def video_to_frames(video_path):
    try:
        if not os.path.exists(curr_path + "/frames"):
            os.makedirs(curr_path + "/frames")
    except OSError:
        print ('Error: Creating directory of data')

    # print(os.path.exists(video_path))
    frame_count=1

    vid=cv2.VideoCapture(video_path)
    frame_cnt = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    image_list = []
            
    while(True):
        ret, frame = vid.read()
        if ret :
            if (frame_count<frame_cnt and frame_count%(frame_cnt//nframe)==0):
                frame_path = curr_path + '/frames' + '/' + str(frame_count//(frame_cnt//nframe)) + '.jpg'
                # print("YES")
                cv2.imwrite(frame_path, frame)
                image_list.append(frame_path)
            frame_count+=1
        else :
            break
        
    vid.release()
    cv2.destroyAllWindows()

    return image_list

class Feat(nn.Module):
    def __init__(self):
        super(Feat, self).__init__()

        self.vgg = models.vgg19(pretrained=True).features.eval().to(device)

        for param in self.vgg.parameters():
            param.requires_grad = False

        self.preprocess = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])

        self.avgpool = nn.AvgPool2d((7,7))

    def forward(self, input_image):
        out = self.preprocess(input_image)
        # print(out.shape)
        out = out.to(device)
        out = out.unsqueeze(0)
        # print(out.shape)
        out = self.vgg(out)
        # print(out.shape)
        out = self.avgpool(out)
        # print(out.shape)
        feat = out.reshape(-1)
        # print(feat.shape)

        return feat

def load_image(img):
    img = Image.open(img).convert('RGB')
    img = img.resize((224, 224), Image.ANTIALIAS)
    return img

def helper(model, video):
    image_list = video_to_frames(video)

    features = torch.zeros((16, 512), requires_grad=False)

    for i in range(len(image_list)):
        if(i==16):
            break
        img = load_image(image_list[i])
        features[i] = model(img)

    return features


def extractFeatures(video_folder_path):
    model = Feat().to(device)

    try:
        if not os.path.exists(video_folder_path + "/feat"):
            os.makedirs(video_folder_path + "/feat")
    except OSError:
        print ('Error: Creating directory of data')

    curr = video_folder_path+ "/video"

    till = 0
    for video in os.listdir(curr):
        # print(os.path.exists(video_folder_path+ "/video"))
        print(f"Preprocessing {video}, Till now: {till}")
        outFilePath = video_folder_path + "/" + "feat" + "/" + video.split(".")[0] + ".pt"
        features = helper(model, curr+ "/" + video)
        # print(features.shape)
        torch.save(features, outFilePath)
        till+=1


video_folder_path = curr_path + "/training_data"
# print(os.path.exists(video_folder_path))
# print(os.path.exists(video_folder_path+ "/video"))
extractFeatures(video_folder_path)