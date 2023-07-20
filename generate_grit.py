'''
@TranRick 2023/07/15
This code use to Generate the Visual Instruction dataset
Section 1 : Taking the GRIT Model to generate the Dense Captioning
Section 2 : Instruction GPT --> Generate the Abstract Visual Instruction

'''


from vision_language_model.GRiT.dense_caption_raw import image_caption_api
import os
import sys
import torch 
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  
from llm_gpt import set_openai_key
import openai
from torchvision import transforms
from utils import CityscapesSegmentation
sys.path.append("/data/rick/autonomous_instruction_dataengine/Autonomous_Driving_Visual_Instruction_DataEngine/vision_language_model/")

#***************  Section 1 GRIT Model to Generate Dense Captioning *****************#
device = "cuda" if torch.cuda.is_available() else "cpu"

class DenseCaptioning():
    def __init__(self, device):
        self.device = device

    def initialize_model(self):
        pass
    
    def image_dense_caption(self, image_src,img_size=384, save_img_path=None, image_name="test.png", save_image=True):
        dense_caption, _, visualized_output, img_shape = image_caption_api(image_src, self.device,image_size=img_size)
        print('\033[1;35m' + '*' * 100 + '\033[0m')
        print("Step2, Dense Caption:\n")
        print(dense_caption)
        print('\033[1;35m' + '*' * 100 + '\033[0m')
        if save_image: 
            if not os.path.exists(save_img_path):
                    os.mkdir(save_img_path)
            out_filename = os.path.join(save_img_path, image_name)
            visualized_output.save(out_filename)
        return dense_caption, visualized_output, img_shape



#***************  Section 2 Instruction GPT --> Generate the Abstract Visual Instruction *****************#
set_openai_key() ## Remember to "export OPENAI_API_KEY=yourkey" in the terminal
# Valid of Select GPT Model
VALID_CHATGPT_MODELS = ['gpt-4', "gpt-35-turbo"]

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
def call_gpt(chatgpt_messages, max_tokens=1024, model="gpt-35-turbo"):
    response = openai.ChatCompletion.create(engine=model, messages=chatgpt_messages,
    temperature=0.7,
    max_tokens=max_tokens,
    top_p=0.95,
    frequency_penalty=1.2,
    presence_penalty=0,
    stop=None)
    reply = response['choices'][0]['message']['content']
    total_tokens = response['usage']['total_tokens']
    return reply, total_tokens

def main(image_path, GPT_model, device, save_path): 
    
    ## Generate Dense Captioning Visual Information
    dense_caption= DenseCaptioning(device)
    name_start = str(image_path).find("/Cityscapes/")
    image_path_ = image_path[name_start-1:]
    dense_caption, _, image_shape= dense_caption.image_dense_caption(image_path,img_size=384, save_img_path=save_path, image_name="test.png", save_image=True)
    print("this is the dense caption", dense_caption)
    print("this is the image shape", image_shape)


    # Generate the Visual Instruction   
    if GPT_model == 'chatgpt':
        GPT_model = 'gpt-35-turbo'
    elif 'gpt4' in GPT_model:
        GPT_model = 'gpt-4'
#     visual_abstract_instruction_prompt=f"Imagine you are a blind but intelligent image captioner, image reasoning understanding who is only given the bounding box and description of each object in a scene.\ 
# Note the provided positions are the object left-top corner coordinates and object sizes image shape is {image_shape}.\ 
# Write the visual description this scene using the relative positions and sizes of objects as follow :\"

    messages_1=[
    {"role": "system", "content": f"Imagine yourself as an intelligent image captioner possessing the unique ability to comprehend object relationships and image reasoning, despite lacking sight. Your task is to describe a scene by utilizing the bounding box and description of each object. Keep in mind that the provided positions indicate the left-top corner coordinates of the objects, and the image has a shape of {image_shape}."},
    {"role": "user", "content": f"Using the relative positions and sizes of the objects provided below, \n{(dense_caption)} please craft a visual description of the scene:"},
    ]
    messages_2=[
    {"role": "system", "content": f"Envision yourself as a highly intelligent in visual perception who excels in understanding object relationships and visual reasoning, even without the sense of sight. Your task is to describe a scene solely based on the bounding box and description of each object. Please note that the provided positions indicate the left-top corner coordinates of the objects, and the image has a shape of {image_shape}."},
    {"role": "user", "content": f"Utilizing the relative positions and sizes of the objects provided as follow: {dense_caption}, kindly compose a visual description, deep scene understanding, visual reasoning and others aspects to extract rich visual information for Driving objectives:\n"},
    ]

    messages_3=[
    {"role": "system", "content": f"Picture yourself as a highly perceptive image captioner with a profound understanding of object relationships and image reasoning, despite lacking the sense of sight. Your task is to describe a scene based solely on the bounding box and description of each object. Please note that the provided positions represent the object's left-top corner coordinates, and the image has a shape of {image_shape}."},
    {"role": "user", "content": f"Please describe the scene using the relative positions and sizes of the objects provided below:\n{dense_caption}"},
    ]


    messages_4=[
    {"role": "system", "content": f"Imagine you possess the remarkable ability to comprehend object relationships and perception reasoning capability, despite being blind. Your task is to describe a scene by utilizing the bounding box and description of each object. Take note that the provided positions correspond to the left-top corner coordinates of the objects, and the image has a shape of {image_shape}."},
    {"role": "user", "content": f"Based on the objects' relative positions and sizes provided as follow: {dense_caption}, please compose a visual description, deep scene understanding, visual reasoning and others aspects to extract rich visual information without providing coordinate in detailed for Driving objective:\n"},
    ]


    ## Current Testing Recommend messages_2 and Messages_4 for Information With Coordinate and Without Coordinate Visual abstract information
    ## If using Coordinate information, try to make sure this Image Shape will be the Final Input Image Shape for the Driving Model
    visual_description=call_gpt(chatgpt_messages=messages_4,  max_tokens=512,model=GPT_model)
    
    return visual_description

if __name__ == '__main__':

    image_path= "/data1/dataset/Cityscapes/leftImg8bit/train/jena/jena_000078_000019_leftImg8bit.png"
    save_path = "/data/rick/autonomous_instruction_dataengine/Autonomous_Driving_Visual_Instruction_DataEngine/Test_images/cityscape_image_tests/grit_model_test"
    GPT_model = 'gpt-4'
    img_name="Dense_caption_T_06_jena_000078_000019_leftImg8bit.png"
    #
    visual_description=main(image_path, GPT_model, device, save_path)
    print(visual_description)