# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 13:30:06 2022

@author: 11320
"""
import numpy as np
import scipy.io as sio
import zarr
import os

# animal = ['893','959','882','910','900','970','38','65','983','955','886']
# state = ['Estrus','Diestrus','Male']
# sess = [['4'],['2'],['4'],['6'],['10'],['2','3'],['1'],['1'],['1'],['3'],['2']]
animal = ['65']
state = ['Male']
sess = [['1'], ['1']]
for ii in range(np.size(animal)):
    for jj in range(np.size(state)):
        session = sess[ii]
        for kk in range(np.size(session)):
            dpath = 'J:/MJH/SortMS_EsDi/M'+animal[ii]+'/'+state[jj]+'/Sess'+session[kk]
            if os.path.exists(dpath):   

                print('---- Find',dpath,', Now Processing ----')
                SavePath = dpath+'Res'
                intpath = dpath+'/minian_intermediate'
                Varray = zarr.open(intpath+'/Y_fm_chk.zarr')
                out = Varray['Y_fm_chk']
                outarray = out.astype(np.uint8)
                #sio.savemat(SavePath+'/varr.mat',{'array':outarray[:]})
                Shape = np.shape(outarray[:])
                length = Shape[0]
                split_mat = np.arange(0,length,9000)
                for idx in range(np.size(split_mat)-1):
                    print('Now Working on',idx,'th mat')
                    start = split_mat[idx]
                    end = split_mat[idx+1] - 1
                    sio.savemat(SavePath+'/varr'+str(idx+1)+'.mat',{'array':outarray[start:end,:,:]})
                sio.savemat(SavePath+'/varr'+str(idx+2)+'.mat',{'array':outarray[split_mat[idx+1]:length,:,:]})    
                    
                print('----',dpath,'Done! ----')
            else:
                print('----',dpath,'Not Find!!!! ----')