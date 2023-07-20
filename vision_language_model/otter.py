'''
TranNhiem 2023/07/17 
This code features used for Otter Visual Instruction Model
+ The GPT Instruct to ask the question --> Otter Model will try to answer --> Extract the abstract Visual Information 
'''

import mimetypes
import os
from io import BytesIO
from typing import Union
import cv2
import requests
import torch
import transformers
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from tqdm import tqdm
import sys
sys.path.append("../..") 
sys.path.append("/data/rick/autonomous_instruction_dataengine/Autonomous_Driving_Visual_Instruction_DataEngine/vision_language_model/Otter/otter")
# sys.path.insert(0, '/data/rick/autonomous_instruction_dataengine/Autonomous_Driving_Visual_Instruction_DataEngine/vision_language_model/Otter/otter/')
from otter.modeling_otter import OtterForConditionalGeneration
