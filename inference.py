'''
@kinanmartin 2023/07/10

AI Driving Assistant
Inference
'''

import argparse
import json
import os
import sys

from PIL import Image
import torch
import yaml
from blips import Viusal_Understanding
# from llm_gpt import (Generate_instruction_Input_output, call_chatgpt,
#                      call_gpt3, get_instructions, prepare_chatgpt_message,
#                      prepare_gpt_prompt, set_openai_key)
# from torchvision import transforms
from utils import (resize_long_edge, CityscapesSegmentation)

from typing import List

# set_openai_key() ## Remember to "export OPENAI_API_KEY=yourkey" in the terminal

##***************************************************************************************************
## ------------  Section 1 Setting the Parameters and Configure BLIP Model -----------------
##***************************************************************************************************
  
def get_blip_model(device='cuda', base_model="blip2", blip_model="OPT2.7B COCO" , cache_dir="/media/rick/f7a9be3d-25cd-45d6-b503-7cb8bd32dbd5/pretrained_weights/BLIP2/", load_bit=4):
    
    BLIP2_models_available = ['FlanT5 XXL','FlanT5 XL COCO','OPT6.7B COCO','OPT2.7B COCO', 'FlanT5 XL','OPT6.7B', 'OPT2.7B',]
    InstructBLIP_available= ['Ins_FlanT5 XXL','Ins_FlanT5 XL','vicuna-13B', 'vicuna-7B',]

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if base_model == "blip2":
        if blip_model in BLIP2_models_available:
            blip_model = Viusal_Understanding(device, base_model=base_model,blip_model=blip_model, cache_dir=cache_dir, load_bit=load_bit)
    elif base_model == "instructblip":
        if blip_model in InstructBLIP_available:
            blip_model = Viusal_Understanding(device, base_model=base_model,blip_model=blip_model, cache_dir=cache_dir, load_bit=load_bit)
    else:
        raise ValueError("Currently only support blip2 and instructblip model")    
        
    return blip_model

# ## Some Default Parameters FOR BLIP llm Model
# BLIP_llm_decoding_strategy="nucleus"
# BLIP_max_length_token_output=100 

def make_instruction_input(base_question=None, driver_question=None, direction=None, gps=None, driver_info=None):
    '''
    Make the instruction prompt input to the BLIP model based on if there is a specified
        driver question, camera direction, gps, or other driver info.
    '''
    out = ''
    if direction is not None:
        out += f"This is a picture from the {direction} camera of a car. "
    else:
        out += "This is a picture from the front camera of a car. "
    out += "You are a helpful driving assistant. "
    if gps is not None:
        out += f"The car is located at {gps}. "
    if driver_info is not None:
        out += f"Here is some information about the driver: {driver_info}. "
    if driver_question is not None:
        out += f"The driver has asked you the following question: '{driver_question}'. "
        out += f"Please give the driver a helpful, detailed answer: "
    else:
        out += f"Please give the driver a useful suggestion or warning: "
        
    print("Prompt to BLIP:")
    print(out)
    print()
    return out

def main(img_path_List: List[str], # path of image 
         instruction_input_List: List[str],
         device, base_model, blip_model, cache_dir, load_bit):
    
    visual_understanding_model = get_blip_model(device=device, 
                                                base_model=base_model, 
                                                blip_model=blip_model, 
                                                load_bit=load_bit,  
                                                cache_dir=cache_dir,)
    responses = {}
    for img_path, instruction_input in zip(img_path_List, instruction_input_List):
        image = Image.open(img_path)
        image = resize_long_edge(image, 224)
        response = visual_understanding_model.abstract_visual_output(image, instruction_input)
        print(response)
        responses[img_path] = response

    return responses

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_model = "instructblip" # ['instructblip' , 'blip2' ]
    blip_model = "vicuna-7B" # ["vicuna-7B", "OPT2.7B COCO", "OPT6.7B COCO",] 
    cache_dir = '/data/rick/pretrained_weights/Instruct_blip/'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    load_bit=4
    
    # instruction_input = "This is a picture from the front-view camera of a car. \
    # You are a helpful driving assistant. Please give the driver of the car a \
    # useful suggestion or warning:"#"Describe this image in detail: " # "Describe this image in detail", "A photo of"

    img_path_List = []
    img_path = "/data/kinan/driving_assistant/Driving_Assistant/other_test_images/4cam_tests/LINE_ALBUM_AttractionPlaceTesting_230707_15.jpg"
    [img_path_List.append(img_path) for i in range(6)]

    instruction_input_List = []
    instruction_input = make_instruction_input(direction='left')
    
    # instruction_input = make_instruction_input(driver_question='How can I get to my destination faster?', 
    #                                            direction='left')
    [instruction_input_List.append(instruction_input) for i in range(6)]
    
    print("BLIP response:\n")
    main(img_path_List,
         instruction_input_List,
         device, base_model, blip_model, cache_dir, load_bit)
    print()
