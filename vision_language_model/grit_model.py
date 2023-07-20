'''
TranNhiem 2023/07/12 
This Code features 
+ Abstract Object Detection & Localization via Dense captioning information 
GRIT --> Image Dense Caption 

'''

import os

from GRiT.dense_caption_raw import image_caption_api
import sys 
from detectron2.utils.logger import setup_logger
sys.path.append( '/data/rick/autonomous_instruction_dataengine/Autonomous_Driving_Visual_Instruction_DataEngine/vision_language_model/GRiT/')

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



## Unit Test 
if __name__ == '__main__':

    image_path= "/data1/dataset/Cityscapes/leftImg8bit/train/jena/jena_000078_000019_leftImg8bit.png"
    save_img_path = "/data/rick/autonomous_instruction_dataengine/Autonomous_Driving_Visual_Instruction_DataEngine/Test_images/cityscape_image_tests/grit_model_test"
    device = "cuda:1"
    img_name="Dense_caption_T_06_jena_000078_000019_leftImg8bit.png"
    dense_caption = DenseCaptioning(device)
    dense_caption, _, image_shape= dense_caption.image_dense_caption(image_path, save_img_path=save_img_path, image_name=img_name)
    print(dense_caption)