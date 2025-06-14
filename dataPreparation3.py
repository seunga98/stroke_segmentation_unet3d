import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
 
import nibabel
import nibabel.processing
import multiprocessing
import scipy.ndimage.morphology


class AphasiaDWIData(Dataset):
    __train_sbjlist = [] # includes subject image path
    __train_data = [] # includes image data with nibabel format : dimension (list:sbj) x (list:modality)
    __train_target = [] # includes label data with nibabel format: dimension (list:sbj) x (1)
    __train_ref_filepath = [] # includes label image filepath, it is used for saving the output image

    __test_sbjlist = []
    __test_data = []
    __test_target = []
    __test_ref_filepath = []

    def __init__(self, modality, path_dataset):

        self.PATH_DATASET = path_dataset
        self.modality = modality
        self.train = True

        print('\t- Data Used : ', self.PATH_DATASET)
        print('\t- MODALITY = ', self.modality)

        with open("/media/hdd_sda/seg_seunga/./test_list_adcb0b1000.txt") as f:
            for line in f:
                line = line.rstrip('\n')
                self.__train_sbjlist.append(line)

        self.__train_data, self.__train_ref_filepath = self.preprocessing(self.__train_sbjlist)
        print('Loaded %d samples'%len(self.__train_ref_filepath))


    def __len__(self):
        if self.train:
            return len(self.__train_sbjlist)
        else:
            return len(self.__test_sbjlist)

    def __getitem__(self, index):
        # Transform nibabel.Nifti1 to torch.Tensor
        # img = torch.from_numpy(np.stack(imglist)).float()
        # img = img.permute(0, 1, 2, 3)  # C x X x Y x Z
        if self.train:
            list_sbj_data = self.__train_data[index]
            do_transform = True
        else:
            list_sbj_data = self.__test_data[index]
            do_transform = False

        img = self.get_image(list_sbj_data, transforms=do_transform)

        return img

    def get_dataset(self):
        if self.train:
            list_data = self.__train_data
        else:
            list_data = self.__test_data

        list_data_as_numpy = []
        for index in range(len(list_data)):
            list_sbj_data = list_data[index]
            img, label = self.get_image(list_sbj_data, transforms=False)
            list_data_as_numpy.append(img)

        data = torch.stack(list_data_as_numpy, dim=0)

        return data

    def get_image(self, list_data, transforms=False):
        """ Here, organize data format and apply transforms(e.g., random rotation, translation, etc)"""
        list_data_as_numpy = []
        for nb_vol in list_data:
            """ Here, apply transforms for input data"""
            # if transforms:
            #     print('random rotation, translation, add noise')
            list_data_as_numpy.append(nb_vol.get_data())

        img = torch.from_numpy(np.stack(list_data_as_numpy)).float()
        img = img.permute(0, 1, 2, 3)  # Channel x X x Y x Z

        return img

    def export_image2(self, PATH_SAVE, res_as_tensor, epoch, index):
        if self.train:
            list_sbjlist = self.__train_sbjlist
            list_data = self.__train_data
            list_reffile = self.__train_ref_filepath
        else:
            list_sbjlist = self.__test_sbjlist
            list_data = self.__test_data
            list_reffile = self.__test_ref_filepath

        nb_refvol = list_data[index][0]

        """Save all the output channels """
        fn_ref = list_reffile[index]
        s, channel, xdim, ydim, zdim = res_as_tensor.size()
        for t in range(channel-1):
            # t = 0
            nb_vol = nibabel.Nifti1Image(res_as_tensor[0, t, :, :, :].data.numpy(), nb_refvol.affine)
            fn_save = os.path.join(PATH_SAVE, list_sbjlist[index],
                                   'export_lesionmask.nii.gz')
            self.img_recon_for_save2(nb_vol, fn_out=fn_save)

    def img_recon_for_save2(self, img_out, fn_out=None):
        # processing -> original

        if fn_out is not None:
            path_out = os.path.dirname(fn_out)
            if os.path.exists(path_out) is not True:
                os.makedirs(path_out)

            nibabel.save(img_out, fn_out)
    ##############################################################################################################################
    # def for preprocessing
    def preprocessing(self, sbjlist):
        print('Starting Preprocessing')
        pool = multiprocessing.Pool(processes=12)
        ret = pool.map(self.preprocessing_subject, sbjlist)
        pool.close()
        pool.join()
        print('Finished Preprocessing')

        list_sbj_data = []
        list_sbj_ref_file = []
        for i in range(len(sbjlist)):
            sbj_imglist, sbj_label_filepath = ret[i]
            list_sbj_data.append(sbj_imglist)
            list_sbj_ref_file.append(sbj_label_filepath)

        return list_sbj_data, list_sbj_ref_file

    def preprocessing_subject(self, sbjid):
        # Load images
        imglist = []
        for surfix in self.modality:
            fn_org = os.path.join(self.PATH_DATASET, sbjid, '%s.nii.gz' % surfix)
            img = self.preprocessing_image(fn_org)
            imglist.append(img)

        return imglist, fn_org

    def preprocessing_image(self, fn_org):
        img_org = nibabel.load(fn_org)
        img_out = img_org

        return img_out

    ##############################################################################################################################
