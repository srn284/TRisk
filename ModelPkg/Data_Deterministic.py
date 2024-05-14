import torch.nn as nn
import torch
import torch.nn as nn
import SurvExp.pytorch_pretrained_bert as Bert
import numpy as np
import copy
import torch.nn.functional as F
import math
from torch.utils.data.dataset import Dataset
import random
import numpy as np
import torch

from HORIZON.CVmortalityPred_inHF.ModelPkg.utils import *
from HORIZON.CVmortalityPred_inHF.ModelPkg.CPHloss import *



# data for var autoencoder deep unsup learning


class TBEHRT_data_formation(Dataset):
    def __init__(self, token2idx, dataframe, max_len, max_age=110, yvocab=None, age_symbol=None, TestFlag=None, binaryF=False, noMLM=True, year = False, expColumn = None, outcomeColumn = None,  cut_chunks=200, breakIndx=100, cut=1000, list2avoid = None):

        if list2avoid is None:
            self.acceptableVoc = token2idx
        else:
            self.acceptableVoc = {x: y for x, y in token2idx.items() if x not in list2avoid}
            print("old Vocab size: ", len(token2idx), ", and new Vocab size: ", len(self.acceptableVoc))
        self.vocab = token2idx
        self.max_len = max_len
        self.code = dataframe.code
        self.age = dataframe.age
        self.year = dataframe.year
        if outcomeColumn is None:
            self.label = dataframe.deathLabel
        else:
            self.label = dataframe[outcomeColumn]
        self.age2idx, _ = age_vocab(max_age, year, symbol=age_symbol)
        self.gender = dataframe.gender
        self.region = dataframe.region
        if expColumn is None:
            self.treatmentLabel = dataframe.diseaseLabel
        else:
            self.treatmentLabel = dataframe[expColumn]
        self.cut =cut
        self.binaryF = binaryF
        self.TestFlag = TestFlag
        self.noMLM = noMLM
        self.year2idx = yvocab
        self.codeS = dataframe.smoke
        self.cut_chunks = cut_chunks
        self.breakindx = breakIndx



    def __getitem__(self, index):
        """
        return: age, code, position, segmentation, mask, label
        """

        # extract data

        age = self.age[index]

        code = self.code[index]
        year = self.year[index]

        gender = int(self.gender[index])
        region = int(self.region[index])
        # extract data
        age = age[(-self.max_len + 1):]
        code = code[(-self.max_len + 1):]
        year = year[(-self.max_len + 1):]
        # treatmentOutcome = 0

        # if self.treatmentLabel[index]!=0:
        #     treatmentOutcome = 1

        treatmentOutcome = torch.LongTensor([ self.treatmentLabel[index]])
        #             treatmentOutcome = self.treatmentLabel_token2idx[self.treatmentLabel[index]]

        # avoid data cut with first element to be 'SEP'
        labelOutcome = self.label[index]

        # if code[0] == 'CLS':
        #     code[0] = 'SEP'

        #
        # else:
        #     code[0] = 'SEP'
        code[-1]= 'CLS'

        mask = np.ones(self.max_len)
        mask[:-len(code)] = 0
        mask = np.append(np.array([1]), mask)



        gender = int(gender)
        # gender = np.repeat(gender, len(code))

        region = int(region)
        # region = np.repeat(region, len(code))

        tokensReal, code2 = code2index(code, self.vocab)
        # pad age sequence and code sequence
        year = seq_padding_reverse(year, self.max_len, token2idx=self.year2idx )

        age = seq_padding_reverse(age, self.max_len, token2idx=self.age2idx)

        if self.noMLM == True:
            tokens, codeMLM, labelMLM = newrandom_mask_cardio3(code, self.vocab)
        else:
            tokens, codeMLM, labelMLM = randommaskreal(code, self.acceptableVoc)

        # tokens, codeMLM, labelMLM = newrandom_mask_cardio3(code, self.vocab)
        # else:
        #     if self.noMLM == False:
        #         tokens, codeMLM, labelMLM = random_mask(code, self.vocab)

        # get position code and segment code
        tokens = seq_padding_reverse(tokens, self.max_len)
        position = position_idx(tokens)
        segment = index_seg(tokens)

        # pad code and label
        # gender = seq_padding_reverse(gender, self.max_len, symbol=self.vocab['PAD'])

        # region = seq_padding_reverse(region, self.max_len, symbol=self.vocab['PAD'])
        # pad code and label
        code2 = seq_padding_reverse(code2, self.max_len, symbol=self.vocab['PAD'])

        codeMLM = seq_padding_reverse(codeMLM, self.max_len, symbol=self.vocab['PAD'])
        labelMLM = seq_padding_reverse(labelMLM, self.max_len, symbol=-1)
        # outCodeS = seq_padding_reverse(outCodeS, self.max_len, symbol=0)

        outCodeS=int(self.codeS[index])


        fixedcovar = np.array([outCodeS, region, gender])
        labelcovar = np.array([-1,-1,-1])
        lenfixedcovar = len(fixedcovar)
        if self.noMLM == False:
            fixedcovar, labelcovar = covarUnsupMaker(fixedcovar)
        code2 = np.append(fixedcovar, code2)
        codeMLM = np.append(fixedcovar, codeMLM)
        # labelMLM = np.append(np.array([-1]*lenfixedcovar), labelMLM)



        # print()
        return torch.LongTensor(age), torch.LongTensor(code2), torch.LongTensor(codeMLM), torch.LongTensor(
            position), torch.LongTensor(segment), torch.LongTensor(year), \
               torch.LongTensor(mask), torch.LongTensor(labelMLM), torch.LongTensor([labelOutcome]), treatmentOutcome, torch.LongTensor([0]),torch.LongTensor([0]) , torch.LongTensor([0]), torch.LongTensor(labelcovar)

    def __len__(self):
        return len(self.code)
    
class Binary_DeepSurvDataset(Dataset):

#     def __init__(self, mnist_dataset, time, event):
#         self.mnist_dataset = mnist_dataset
#         self.time, self.event = tt.tuplefy(time, event).to_tensor()
    def __init__(self, token2idx, dataframe, max_len, max_age=110, year=False, age_symbol=None, min_visit=5):
        # dataframe preproecssing
        # filter out the patient with number of visits less than min_visit
        self.vocab = token2idx
        self.max_len = max_len
        self.code = dataframe.code
        self.age = dataframe.age
#         self.label = dataframe.label
        self.age2idx, _ = age_vocab(max_age, year, symbol=age_symbol)
        self.gender = 0
        self.region = 0
        self.time2event = dataframe.time2event
        self.label = dataframe.label
        self.baselineage = dataframe.baselineage
#         self.time2event, self.label = tt.tuplefy(dataframe.time2event.values, dataframe.label.values).to_tensor()

    def __len__(self):
        return len(self.code)
    
    
    def __getitem__(self, index):
        
        age = self.age[index]
        bage = self.baselineage[index]
        code = self.code[index]
        label = self.label[index]
        gender = int(self.gender)
        region = int(self.region)
        # extract data
        age = age[(-self.max_len + 1):]
        code = code[(-self.max_len + 1):]

        # avoid data cut with first element to be 'SEP'
        if code[0] != 'SEP':
            code = np.append(np.array(['CLS']), code)
            age = np.append(np.array(str(bage)), age)
        else:
            code[0] = 'CLS'
            age[0] = str(bage)

        # mask 0:len(code) to 1, padding to be 0
        mask = np.ones(self.max_len)
        mask[len(code):] = 0

        # pad age sequence and code sequence
        age = seq_padding(age, self.max_len, token2idx=self.age2idx)

        tokens, code = code2index(code, self.vocab)
        #         _, label = code2index(label, self.vocab)

        # get position code and segment code
        tokens = seq_padding(tokens, self.max_len)
        position = position_idx(tokens)
        segment = index_seg(tokens)
        position[0] = position[-1]+1
        segment[0] = 1

        # pad code and label
        code = seq_padding(code, self.max_len, symbol=self.vocab['PAD'])
        #         label = seq_padding(label, self.max_len, symbol=-1)
#         print(code)
#         print(torch.LongTensor(age), torch.LongTensor(code), torch.LongTensor(position), torch.LongTensor(segment), \
#                torch.LongTensor(mask), torch.LongTensor(self.time2event[index]), torch.LongTensor( self.label[index])
        
        
        return torch.LongTensor(age), torch.LongTensor(code), torch.LongTensor(position), torch.LongTensor(segment), \
               torch.LongTensor(mask), torch.FloatTensor([self.time2event[index]]), torch.LongTensor( [self.label[index]]),torch.FloatTensor( [self.label[index]])
        
        

