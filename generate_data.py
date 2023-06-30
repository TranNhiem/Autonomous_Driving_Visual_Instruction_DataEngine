
##********************************************************************************************
## Step 3 Generate Caption ChatGPT & BLIP2 (Task Instructed Instructed)
##********************************************************************************************

import sys
sys.path.append('/home/rick/DataEngine_Pro/Boostraping_Multi_modalities_SSL/DataModule/ChatCaptioner')
#sys.path.append('/home/twshymy868/SSL_DataEngine/Boostraping_Multi_modalities_SSL/DataModule/ChatCaptioner')
import yaml
import torch

from chatcaptioner.chat import set_openai_key, caption_images, get_instructions
from chatcaptioner.blip2 import Blip2
from chatcaptioner.utils import RandomSampledDataset, plot_img, print_info
# openai.api_type = "azure"
openai.api_base = "https://sslgroupservice.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai_key = os.environ["OPENAI_API_KEY"]

# CUDA_VISIBLE_DEVICES="1"
# ## export OPENAI_API_KEY=0aee54a3f2df4c55aea57bf3cf2e99a6 #97f22a7a32ff4ff4902003896f247ca2
# set_openai_key(openai_key)
# ## Adding These Line of code in Chat.py if using Azure OpenAI

'''
    VALID_CHATGPT_MODELS = ['gpt-3.5-turbo', "gpt-35-turbo"]## lINE 54
    from Line 74-> 77
    openai.api_type = "azure"
    openai.api_version = "2023-03-15-preview" 
    openai.api_base = "https://sslgroupservice.openai.azure.com/"#os.getenv("OPENAI_API_BASE")  # Your Azure OpenAI resource's endpoint value.
    openai.api_key = os.getenv("OPENAI_API_KEY")
    question_model_tag ="gpt-35-turbo"
''' 

    
device = "cuda" if torch.cuda.is_available() else "cpu"

blip2s = {
   #'FlanT5 XXL': Blip2('FlanT5 XXL', device_id=0, bit8=True), # load BLIP-2 FlanT5 XXL to GPU0. Too large, need 8 bit. About 20GB GPU Memory
    'OPT2.7B COCO': Blip2('OPT2.7B COCO', device_id=0, bit8=False), # load BLIP-2 OPT2.7B COCO to GPU1. About 10GB GPU Memory
    #'FlanT5 XL COCO': Blip2('FlanT5 XL COCO', device_id=0, bit8=False), # load BLIP-2 OPT2.7B COCO to GPU1. About 10GB GPU Memory
    #'OPT6.7B COCO': Blip2('OPT6.7B COCO', device_id=2, bit8=True), # load BLIP-2 OPT6.7B COCO to GPU2. Too large, need 8 bit.
}

blip2s_q = {}

## ------------ Setting the Parameters -----------------

# set the dataset to test
dataset_name = 'cityscape_train_coarse_OPT_2_7_new'  # current options: 'artemis', 'cc_val', 'coco_val'
# set the number of chat rounds between GPT3 and BLIP-2
n_rounds = 8
# set the number of visible chat rounds to BLIP-2. <0 means all the chat histories are visible.
n_blip2_context = 1
# if print the chat out in the testing
print_chat = True
# set the question model
question_model_tag ="gpt-35-turbo"## for OPENAI "gpt-3.5-turbo"

# ------------ Loading the Dataset -----------------
## preparing the folder to save results

#SAVE_PATH = '/home/twshymy868/city_scape_synthetic/{}/{}'.format(question_model_tag, dataset_name)
SAVE_PATH = '/media/rick/f7a9be3d-25cd-45d6-b503-7cb8bd32dbd5/cityscape_synthetic/{}/{}'.format(question_model_tag, dataset_name)

if not os.path.exists(SAVE_PATH):
    os.makedirs(os.path.join(SAVE_PATH, 'caption_result'))
with open(os.path.join(SAVE_PATH, 'instruction.yaml'), 'w') as f:
    yaml.dump(get_instructions(), f)

if question_model_tag in blip2s_q:
    question_model = blip2s_q[question_model_tag]
else:
    question_model = question_model_tag

##------------ Testing Generate Question-----------------
for i, (image, label) in enumerate(dataset):
    sample_img_ids=i#+15000
    #image.save(f'./temp_imgs/original_image_new{i}.png')
    caption_images(blip2s, 
                image, 
                sample_img_ids, 
                save_path=SAVE_PATH, 
                n_rounds=n_rounds, 
                n_blip2_context=n_blip2_context,
                model=question_model,
                print_mode='chat')
    
    if i==5000:
        break 
