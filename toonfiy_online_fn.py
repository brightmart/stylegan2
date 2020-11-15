# -*- coding: utf-8 -*-

# 给定一个图片，生成一个风格化的图片

import pretrained_networks
import os
# use my copy of the blended model to save Doron's download bandwidth
# get the original here https://mega.nz/folder/OtllzJwa#C947mCCdEfMCRTWnDcs4qw
blended_url = "./models/ffhq-cartoon-blended-64.pkl" #"https://drive.google.com/uc?id=1H73TfV5gQ9ot7slSed_l-lim9X7pMRiU"
ffhq_url = "http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-f.pkl"

_, _, Gs_blended = pretrained_networks.load_networks(blended_url)
print("load Gs_blended end...")
_, _, Gs = pretrained_networks.load_networks(ffhq_url)


def align_image(source_image_path):
    """
    找到人的头像
    :return:
    """
    cmd="python align_images.py data/raw data/aligned"
    os.system(cmd)

def find_latent_code():
    """
    找到
    :return:
    """
def toonify_fn(source_image_path,target_image_path):
    """
    给定一个图片，生成一个风格化的图片
    source_image_path: 源图像的位置
    :return: target_image_path 目标图像的位置
    """
    align_image(source_image_path)
    pass


source_image_path='./data/raw/bright-0.jpg'
target_image_path=''
toonify_fn(source_image_path,target_image_path)