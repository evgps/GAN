import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="3" # change 0  with whatever card is available
# os.environ["CUDA_LAUNCH_BLOCKING"]="1"
import numpy as np
from trainVAE_D import trainVAE_D
import torch
torch.set_default_tensor_type(torch.cuda.FloatTensor)
torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.enabled = False
"""
you shuld use this script in this way:
python trainVAE_D.py <epoches> <batch_size> <pretrainD?> <traindatafilename>  <styledatafilename> ganName

for instance: 
python trainVAE_D.py 1000 20 yes/no ./data/trainDataOfIndex.npy ./data/style ./Model/gan.pkl
"""

booldic = {'yes':True,
            'y':True,
            'Y':True,
            'Yes':True,
            'YES':True,
            'no':False,
            'N':False,
            'n':False,
            'NO':False,
            'No':False,}

ds = torch.load('./Model/Ds.pkl').cuda()
ds_emb = torch.load('./Model/Ds_emb.pkl').cuda()

train_data = np.load('./data/trainDataOfIndex.npy')
gan_path = './Model/gan.pkl'
style_path = './data/style'
epoches = 30
batch_size = 50
pretrainD = False

trainVAE_D(epoches, batch_size, train_data, ds, ds_emb,  gan_path, style_path,pretrainD)

print ("finished trainning.......................")