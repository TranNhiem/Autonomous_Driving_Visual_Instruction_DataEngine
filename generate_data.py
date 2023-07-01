'''
@TranRick 2023/06/30
This code use to Generate the Visual Instruction dataset
Section 1 : Setting the Parameters and Configure BLIP Model 
Section 2 : Construct Instruction Input from GPT and Instruction Responese from BLIP --> Suggestions & Solution from GPT 

'''
import sys
import os 
import yaml
import torch
from blips import Viusal_Understanding
from utils import  CityscapesSegmentation, print_info
from llm_gpt import set_openai_key, prepare_gpt_prompt, get_instructions, call_gpt3, call_chatgpt, prepare_chatgpt_message,Generate_instruction_Input_output


##***************************************************************************************************
## ------------  Section 1 Setting the Parameters and Configure BLIP Model -----------------
##***************************************************************************************************
  
device = "cuda" if torch.cuda.is_available() else "cpu"
set_openai_key() ## Remember to "export OPENAI_API_KEY=yourkey" in the terminal

## ------------ Setting the BLIP -----------------
BLIP2_models_available = ['FlanT5 XXL','FlanT5 XL COCO','OPT6.7B COCO','OPT2.7B COCO', 'FlanT5 XL','OPT6.7B', 'OPT2.7B',]
InstructBLIP_available= ['Ins_FlanT5 XXL','Ins_FlanT5 XL','vicuna-13B', 'vicuna-7B',]

base_model = "blip2" # ['instructblip' , 'blip2' ]
blip_model="FlanT5 XXL" 
cache_dir='/media/rick/f7a9be3d-25cd-45d6-b503-7cb8bd32dbd5/pretrained_weights/Instruct_blip/'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
load_bit=4 # [4, 8, 16] #Loading model in Low bit Precision to save memory
blip_model=Viusal_Understanding(device, base_model=base_model,blip_model=blip_model, cache_dir=cache_dir, load_bit=load_bit)

BLIP_llm_decoding_strategy="nucleus"
BLIP_max_length_token_output=100 
##***************************************************************************************************
##  Section 2  onstruct Instruction Input from GPT and Instruction Responese from BLIP --> Suggestions & Solution from GPT -----------------
##***************************************************************************************************
  



## ------------ Instruction input and Responese -----------------
# set the number of question or Instruction input rounds between GPT and BLIP-2
n_rounds = 8
# set the number of visible chat rounds to BLIP-2. <0 means all the chat histories are visible.
n_blip2_context = 1
# if print the chat out in the testing
print_chat = True
# set the question model
question_model_tag ="gpt-4"##  "gpt-35-turbo"

VALID_CHATGPT_MODELS = ['gpt-4', "gpt-35-turbo"]
VALID_GPT3_MODELS = ['text-davinci-003', 'text-davinci-002', 'davinci']

# set the dataset to test
instruction_dataset = 'cityscape_train_coarse_OPT_2b7'  # current options: 'artemis', 'cc_val', 'coco_val'


## 1.1 Setting Prompt to GPTs Model to Create Instruction input for Blip2 or InstructBLIP

input_INSTRUCTION = \
"I have an image. " \
"Ask me questions about the content of this image related to Driving Domain, the information providing is related to street views information. " \
"Carefully asking me informative questions to maximize your information about this image content. " \
"Each time ask one question only without giving an answer. " \
"Avoid asking yes/no questions." \
"I'll put my answer beginning with \"Answer:\"." \


sub_INSTRUCTION = \
"Next Question. Avoid asking yes/no questions. \n" \
"Question: "


## 1.2 Setting Prompt for GPTs Model Provide the Advice and Suggestion 

solution_INSTRUCTION = \
'Now summarize the information you get in a few sentences. ' \
'Based on the summarization you are a helpful assistant please provide some suggestion and advice to Driver. ' \
'Don\'t add information. Don\'t miss information. \n' \
'Summary: '


ANSWER_INSTRUCTION = 'Answer given questions. If you are not sure about the answer, say you don\'t know honestly. Don\'t imagine any contents that are not in the image.'
SUB_ANSWER_INSTRUCTION = 'Answer: '  # template following blip2 huggingface demo
FIRST_instruction = 'Describe this image in detail.'

instruction_dict=get_instructions(input_INSTRUCTION, sub_INSTRUCTION, solution_INSTRUCTION, ANSWER_INSTRUCTION, SUB_ANSWER_INSTRUCTION)

## 1.3 Setting Prompt for GPTs Model Provide the Advice and Suggestion
def summarize_and_suggestion(questions, answers, model, max_gpt_token=100):
    
    if model in VALID_GPT3_MODELS:
        summary_prompt = prepare_gpt_prompt(
                    input_INSTRUCTION, 
                    questions, answers, 
                    solution_INSTRUCTION)
        summary, n_tokens = call_gpt3(summary_prompt, model=model, max_tokens=max_gpt_token)

    elif model in VALID_CHATGPT_MODELS:
        summary_prompt = prepare_chatgpt_message(
                    input_INSTRUCTION, 
                    questions, answers, 
                    solution_INSTRUCTION
                )
        summary, n_tokens = call_chatgpt(summary_prompt, model=model, max_tokens=max_gpt_token)
  
    else:
        raise ValueError('{} is not a valid question model'.format(model))
        
    summary = summary.replace('\n', ' ').strip()
    return summary, summary_prompt, n_tokens


def visual_instruction_input_response(blip2, image, GPT_model, n_rounds=10, max_gpt_token=100, n_blip2_context=-1, print_mode='no'):
    
    if GPT_model == 'gpt3':
        GPT_model = 'text-davinci-003'
    elif GPT_model == 'chatgpt':
        GPT_model = 'gpt-35-turbo'

    
    results = {}
    instruction_input_output = Generate_instruction_Input_output(
                    image, blip2, GPT_model, 
                    FIRST_instruction, input_INSTRUCTION,
                    sub_INSTRUCTION, 
                    VALID_CHATGPT_MODELS, VALID_GPT3_MODELS,
                    ANSWER_INSTRUCTION, SUB_ANSWER_INSTRUCTION,
                    max_gpt_token, n_blip2_context,
                    )

    questions, answers, n_token_chat = instruction_input_output.chatting(n_rounds, print_mode=print_mode, 
                                                                         BLIP_llm_decoding_strategy=BLIP_llm_decoding_strategy, 
                                                                         BLIP_max_length_token_output=BLIP_max_length_token_output)

    summary, summary_prompt, n_token_sum = summarize_and_suggestion(questions, answers, model=GPT_model)

    results['Visual_instruction'] = {'instruction_input_response': summary, 'Suggestion_advice': summary_prompt, 'n_token': n_token_chat + n_token_sum}
    results['BLIP2+OurPrompt'] = {'blip_model_response': answers[0]}

    # Default BLIP2 caption
    caption = blip2.caption(image)
    results['BLIP_model'] = {'blip_model_response': caption}
    
    return results



def generate(blip2s, image, img_ids, model, save_path='', n_rounds=10, n_blip2_context=-1, print_mode='no'):
    """
    Caption images with a set of blip2 models

    Args:
        blip2s (dict): A dict of blip2 models. Key is the blip2 model name
        image: Single image input PIL image 
        img_ids (list): a list of image ids in the dataset used to caption
        model (str or Blip2): the model name used to ask quetion. Valid values are 'gpt3', 'chatgpt', and their concrete model names 
                    including 'text-davinci-003', 'davinci,' and 'gpt-3.5-turbo'.
                    If passing a Blip2 instance, will use its backend LLM.
        save_path (str): the path to save caption results. If it is empty, results are not being saved.
        n_rounds (int): the number of chat rounds
        n_blip2_context (int): how many previous QA rounds can blip2 see. negative value means blip2 can see all 
        print_mode (str): print mode. 'chat' for printing everying. 'bar' for printing everthing but the chat process. 'no' for no printing
    """
    if model == 'gpt3':
        model = 'text-davinci-003'
    elif model == 'chatgpt':
        model = 'gpt-3.5-turbo'
    
    #for img_id in tqdm(img_ids, disable=print_mode!='no'):
    caption_path = os.path.join(save_path, 'caption_result', '{}.yaml'.format(img_ids))
    ## check the weight path if not create the path
    # if not os.path.exists(caption_path):
    #     os.makedirs(caption_path)

    if os.path.exists(caption_path):
        return 
    # if print_mode != 'no':
    #     print('Image ID {}'.format(img_id))
        
    #image, gt_captions = dataset.fetch_img(img_id)


    
    info = {'setting':
                {
                    #'dataset': dataset.name,
                    'id': img_ids,
                    #'GT': {'caption': [caption.replace('\n', ' ').strip() for caption in gt_captions]},
                    'n_rounds': n_rounds
                }
            }

    for blip2_tag, blip2 in blip2s.items():
        info[blip2_tag] = visual_instruction_input_response(blip2, 
                                        image, 
                                        n_rounds=n_rounds, 
                                        n_blip2_context=n_blip2_context, 
                                        model=model,
                                        print_mode=print_mode)

    if print_mode != 'no':
        print_info(info)
        plot_img(image)
    
    if save_path:
        with open(caption_path, 'w') as f:
            yaml.dump(info, f)





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
