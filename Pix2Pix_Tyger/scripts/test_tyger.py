"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import html, util 
import sys
import numpy as np
import torch
from PIL import Image


def pix2pix_knee(dataA,dataB):
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    dataset = create_dataset(opt, dataA, dataB)
    model = create_model(opt)      
    model.setup(opt)           
    for i, data in enumerate(dataset):
        model.set_input(data)  
        model.test()           
        visuals = model.get_current_visuals()  
        for label, im_data in visuals.items():
            if label == 'fake_B':
                im = util.tensor2im(im_data)
                imgRed = (0.2989 * im[:, :, 0] + 
                            0.5870 * im[:, :, 1] + 
                            0.1140 * im[:, :, 2])
                # imgRed = imgRed.astype(np.uint8)
                # imagen_pil = Image.fromarray(imgRed)
                # imagen_pil.save(label+'.png')
                # print(label)
    return imgRed

if __name__ == '__main__':

    dataA = np.load('datasets/test1/achieva/test/RMP_044_Right_14.npy')
    dataB = np.load('datasets/test1/physio/test/RMP_044_Right_14.npy')
    pix2pix_knee(dataA,dataB)