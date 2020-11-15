# -*- coding: utf-8 -*-

# 给定一个图片，生成一个风格化的图片

import pretrained_networks
import os

import numpy as np
from PIL import Image
import dnnlib
import dnnlib.tflib as tflib
from pathlib import Path
# use my copy of the blended model to save Doron's download bandwidth
# get the original here https://mega.nz/folder/OtllzJwa#C947mCCdEfMCRTWnDcs4qw
blended_url = "./models/ffhq-cartoon-blended-64.pkl" #"https://drive.google.com/uc?id=1H73TfV5gQ9ot7slSed_l-lim9X7pMRiU"
ffhq_url = "http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-f.pkl"

_, _, Gs_blended = pretrained_networks.load_networks(blended_url)
print("load Gs_blended end...")
_, _, Gs = pretrained_networks.load_networks(ffhq_url)

# step 1
def align_image():
    """
    找到人的头像
    :return:
    """
    cmd="python align_images.py ./data/raw ./data/aligned"
    os.system(cmd)
    print("step1.align_image.end..")


# step 2:
def find_latent_code():
    """
    找到把图片投影到StyleGAN2的空间
    :return:
    """
    cmd="python project_images.py --num-steps 500 ./data/aligned ./data/generated" # Project real-world images into StyleGAN2 latent space
    os.system(cmd)
    print("step2.find_latent_code.end..")

# step 3:
def blend_latent_code_with_new_model():
    """
    使用blended的模型，利用latent code做图像生成
    :return:
    """
    latent_dir = Path("./data/generated")
    latents = latent_dir.glob("*.npy")
    print("#blend_latent_code_with_new_model.latents:", list(latents))
    for latent_file in latents:
        print("blend_latent_code_with_new_model.latent_file:", latent_file)
        latent = np.load(latent_file)
        latent = np.expand_dims(latent, axis=0)
        synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=False),minibatch_size=8)
        images = Gs_blended.components.synthesis.run(latent, randomize_noise=False, **synthesis_kwargs)
        Image.fromarray(images.transpose((0, 2, 3, 1))[0], 'RGB').save(latent_file.parent / (f"{latent_file.stem}-toon.jpg"))
    print("step3.blend_latent_code_with_new_model.end..")


def toonify_fn():
    """
    给定一个图片，生成一个风格化的图片
    source_image_path: 源图像的位置
    :return: target_image_path 目标图像的位置
    """
    # step 1: 找到人的头像
    align_image()
    # step 2: 找到把图片投影到StyleGAN2的空间
    find_latent_code()
    # step 3: 使用blended的模型，利用latent code做图像生成
    blend_latent_code_with_new_model()

toonify_fn()