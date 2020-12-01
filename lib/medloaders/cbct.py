# -*- coding:utf-8 -*-
from .cbct_utils import *
import torch
from torch.utils.data import Dataset
import lib.utils as utils

class CBCTDataset(Dataset):
    def __init__(self,args,mode,split_id,dataset_path='./datasets'):
        self.mode=mode
        self.root=dataset_path
        self.save_txt_name=os.path.join(dataset_path,'cbct',mode)+'.txt'
        #todo
        self.affine=np.zeros((4,4))
        self.full_volume = [64, 64, 64, 1]
        self.load=args.loadData
        img_npy_list = sorted(os.listdir(os.path.join(dataset_path, 'cbct', 'image')))

        if self.load:
            self.list = utils.load_list(self.save_txt_name)
            return

        if self.mode=='train':
            npy_list=img_npy_list[:split_id]
            self.list=subvolume(dataset_path,npy_list,self.mode)
        elif self.mode=='val':
            npy_list = img_npy_list[split_id:]
            self.list = subvolume(dataset_path, npy_list,self.mode)
            #todo

        save_list(self.save_txt_name,self.list)


    def __getitem__(self, item):
        img_npy_path,lbl_npy_path=self.list[item]
        return torch.FloatTensor(np.load(img_npy_path)),torch.FloatTensor(np.load(lbl_npy_path))

    def __len__(self):
        return len(self.list)








        # self.save_txt_name=self.root+'/'+mode+'.txt'
        # split_percent=args.split
        # if args.dataset_name=='CBCT':
        #     total_data=20
        #     split_idx=int(split_percent*total_data)