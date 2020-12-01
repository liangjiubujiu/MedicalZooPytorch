# -*- coding:utf-8 -*-
import os
import numpy as np
import pydicom
from PIL import Image
import cv2
import pickle
import matplotlib.pyplot as plt
def FillHole(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    len_contour = len(contours)
    if not contours:
        return mask
    contour_list = []
    for i in range(len_contour):
        drawing = np.zeros_like(mask, np.uint8)  # create a black image
        img_contour = cv2.drawContours(drawing, contours, i, (255, 255, 255), -1)
        contour_list.append(img_contour)

    out = sum(contour_list)
    return out

def convert_from_dicom_to_jpg(img, low_window, high_window):
    """

    :param img: dicom图像的像素值信息
    :param low_window: dicom图像像素值的最低值
    :param high_window: dicom图像像素值的最高值
    :param save_path: 新生成的jpg图片的保存路径
    :return:
    """
    lungwin = np.array([low_window * 1., high_window * 1.])  # 将pydicom解析的像素值转换为array
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])  # 将像素值归一化0-1
    newimg = (newimg * 255).astype('uint8')  # 再转换至0-255，且将编码方式由原来的unit16转换为unit8
    # print(newimg.shape)
    return newimg

def generate_gray(img_path,Fixed_RESHAPE_SIZE,mode='png'):


    if mode == 'dcm':
        ds = pydicom.dcmread(img_path)
        # ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        img = np.uint(ds.pixel_array)
        high = np.max(img)  # 找到最大的
        low = np.min(img)  # 找到最小的
        # 调用函数，开始转换
        img = convert_from_dicom_to_jpg(img, low, high)

        img = np.array(Image.fromarray(np.uint8(img)).resize((Fixed_RESHAPE_SIZE, Fixed_RESHAPE_SIZE), Image.ANTIALIAS))
    else:
        img = np.asarray(np.uint8(Image.open(img_path).convert("L").resize((Fixed_RESHAPE_SIZE, Fixed_RESHAPE_SIZE))))


    return img

def generate_img_volume(img_folder,folder,dataset_path,Fixed_RESHAPE_SIZE):
    img_list = []
    for file in img_folder:
        img_path = os.path.join(dataset_path,'image', folder, file)
        img_np = generate_gray(img_path, Fixed_RESHAPE_SIZE, mode='dcm')
        # cv2.imshow('',img_np)
        # cv2.waitKey(1)
        img_list.append(img_np)
    img_npy=np.array(img_list)
    return img_npy

def generate_mask_volume(img_folder,folder,dataset_path,Fixed_RESHAPE_SIZE,hole=True):

        masks=[]
        for file in img_folder:
            file_path = os.path.join(dataset_path,'label', folder, file)
            img_grey = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            img_grey = cv2.resize(img_grey, (Fixed_RESHAPE_SIZE, Fixed_RESHAPE_SIZE))
            if hole:
                mask = FillHole(img_grey)
            else:
                th1 = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9,
                                            2)  # 换行符号 \
                mask = 255 - th1
            mask[mask != 0] == 255
            # cv2.imshow('',mask)
            # cv2.waitKey(100)
            masks.append(mask)
        mask_npy=np.array(masks)
        return mask_npy

def find_no_zero_labels_mask(full_segmentation_map, th_percent, box,num_all,num_crop_all):
    full_segmentation_map[full_segmentation_map > 0] = 1
    num_all_non_zero = full_segmentation_map.sum()
    crop_map = full_segmentation_map[box[0]:box[1], box[2]:box[4], box[3]:box[5]]
    num_crop_non_zero = crop_map.sum()






    thes = num_all_non_zero / num_all
    crop_thes=num_crop_non_zero/num_crop_all

    label_percent = crop_thes/thes

    if label_percent >= th_percent:
        return True
    else:
        return False

    # if num_crop_non_zero==th_percent*num_crop_all:
    #     return True
    # else:
    #     return False


def load_medical_image(img_np, box):
    return img_np[box[0]:box[1], box[2]:box[4], box[3]:box[5]]


def save_list(name, list):
    with open(name, 'wb') as fp:
        pickle.dump(list, fp)


def subvolume(dataset_path,ilist,type):
    dataset_path=os.path.join(dataset_path,'cbct')
    Fixed_RESHAPE_SIZE=512
    th_percent=1#30 for training 10 for val
    box_size=128
    overlap=box_size//2
    npy_list = []
    ilist=sorted(ilist)
    for ifolder in ilist:
        img_list = sorted(os.listdir(os.path.join(dataset_path,'image', ifolder)))
        lal_list = sorted(os.listdir(os.path.join(dataset_path,'label', ifolder)))


        img_np = generate_img_volume(img_list,ifolder,dataset_path,Fixed_RESHAPE_SIZE)
        full_segmentation_map = generate_mask_volume(lal_list,ifolder,dataset_path,Fixed_RESHAPE_SIZE)
        length_vol=img_np.shape[0]
        num=0
        if not os.path.exists(os.path.join(dataset_path,'generated')):
            os.makedirs(os.path.join(dataset_path,'generated'))
        if not os.path.exists(os.path.join(dataset_path,'generated',str(box_size)+'_'+type)):
            os.makedirs(os.path.join(dataset_path,'generated',str(box_size)+'_'+type))
        box = [0,0, 0, 0, 0, 0]
        num_all, num_crop_all=length_vol*Fixed_RESHAPE_SIZE*Fixed_RESHAPE_SIZE,box_size*box_size*box_size
        for length in range(0,length_vol-box_size,overlap):
            for width in range(0,Fixed_RESHAPE_SIZE-box_size,overlap):
                for height in range(0, Fixed_RESHAPE_SIZE - box_size, overlap):

                    box=[length,length+box_size,width,height,width+box_size,height+box_size]
                    if find_no_zero_labels_mask(full_segmentation_map, th_percent, box,num_all, num_crop_all):
                        print('*****************************')
                        img_tensor = load_medical_image(img_np, box)
                        ann_tensor = load_medical_image(full_segmentation_map, box)
                        img_npy_path=os.path.join(dataset_path,'generated',str(box_size)+'_'+type,'id_'+str(ifolder)+'_s_'+str(num)+'.npy')
                        ann_npy_path=os.path.join(dataset_path,'generated',str(box_size)+'_'+type,'id_'+str(ifolder)+'_s_'+str(num)+'_seg.npy')
                        np.save(img_npy_path,img_tensor)
                        np.save(ann_npy_path,ann_tensor)
                        npy_list.append(tuple([img_npy_path,ann_npy_path]))
                    print(num)
                    num = num + 1


    return npy_list

