import numpy as np
import cv2
import pickle

class LoadData:
    '''
    Class to laod the data
    '''
    def __init__(self, data_dir, cached_data_file):
        '''
        :param data_dir: directory where the dataset is kept
        :param cached_data_file: location where cached file has to be stored
        '''
        self.data_dir = data_dir
        self.mean = np.zeros(3, dtype=np.float32)
        self.std = np.zeros(3, dtype=np.float32)
        self.trainImList = list()
        self.valImList = list()
        self.trainDepthList = list()
        self.valDepthList = list()
        self.trainAnnotList = list()
        self.valAnnotList = list()
        self.cached_data_file = cached_data_file

    def readFile(self, fileName, trainStg=False):
        '''
        Function to read the data
        :param fileName: file that stores the image locations
        :param trainStg: if processing training or validation data
        :return: 0 if successful
        '''
        no_files = 0
        with open(self.data_dir + '/' + fileName, 'r') as textFile:
            for line in textFile:
                line_arr = line.split()
                img_file = ((self.data_dir).strip() + '/NJU2K+NLPR/RGB/' + line_arr[0].strip() + ".jpg").strip()
                depth_file = ((self.data_dir).strip() + '/NJU2K+NLPR/depth/' + line_arr[0].strip() + ".bmp").strip()
                label_file = ((self.data_dir).strip() + '/NJU2K+NLPR/GT/' + line_arr[0].strip() + ".png").strip()
                
                if trainStg == True:
                    self.trainImList.append(img_file)
                    self.trainDepthList.append(depth_file)
                    self.trainAnnotList.append(label_file)
                else:
                    self.valImList.append(img_file)
                    self.valDepthList.append(depth_file)
                    self.valAnnotList.append(label_file)
                no_files += 1
        return 0

    def processData(self):
        '''
        main.py calls this function
        We expect train.txt and val.txt files to be inside the data directory.
        :return:
        '''
        print('Processing training data')
        return_train = self.readFile('NJU2K+NLPR_train.txt', trainStg=True)

        print('Processing validation data')
        return_val = self.readFile('NJU2K/NJU2K_test.txt')

        print('Pickling data')
        if return_train == 0 and return_val == 0:
            data_dict = dict()
            data_dict['trainIm'] = self.trainImList
            data_dict['trainDepth'] = self.trainDepthList
            data_dict['trainAnnot'] = self.trainAnnotList
            data_dict['valIm'] = self.valImList
            data_dict['valDepth'] = self.valDepthList
            data_dict['valAnnot'] = self.valAnnotList
            pickle.dump(data_dict, open(self.cached_data_file, "wb"))
            return data_dict
        return None
