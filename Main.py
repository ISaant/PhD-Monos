#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 21:14:49 2021

@author: isaac
"""

import numpy as np
import os 
import scipy.io
from scipy import signal
import pandas as pd
from Fun import *
from PlotCwt import PlotCWT as pltCwt
import matplotlib.pyplot as plt
import pickle
import time
# from CNN import CNN

#Hiperparametros Usuario
path='/home/isaac/Documents/Doctorado CIC/Victor/registros'
Dir=np.sort(os.listdir(path))[1:]
alignEvents = ['manosFijasIni','touchIni','robMovIni','targOn']
endEvents = [['cmdIni','waitCueFin','touchCueIni','manosFijasFin'],
            ['robMovIni','robMovFin','touchCueFin','touchFin',],
            ['targOn','waitRespFin','targOff']]
Columns= [[0, 1, 2, 3, 4],[5,12],[12, 13, 6, 7],[8, 9, 10, 11]]
fs=1000
bF,aF=signal.butter(3,[.1/(fs/2), 150/(fs/2)],btype='bandpass')
b60,a60=signal.butter(4,[59/(fs/2),61/(fs/2)],btype='bandstop')
filterCoeff=[aF,bF,a60,b60]
winsize=256
lev=5

#MFCCs
emphasize=False
nfilt = 60
NFFT = 512
num_ceps='all'
#num_ceps=12
sig_lift=False    
hamming=False
frame_length=winsize
frame_step=winsize

problems=['derVsizq','magnitude','direction']


with open('NonBrokenElectrodes', 'rb') as f: #train.plicke es un archivo donde se guardaron los batch de entrenamiento con 
    NonBrokenElectrodes= pickle.load(f)

colorMatrix=np.array([[255, 0, 0], [255, 77, 0], [255, 197, 0], [50, 182, 4], [36, 128, 39],
                      [36, 128, 109], [0, 255, 170], [0, 243, 255], [0, 116, 255], [36, 53, 128],
                      [0, 39, 255], [124, 0, 255], [220, 0, 255], [255, 0, 228],
                      [121, 36, 128], [255, 0, 73], [40, 91, 91], [143, 138, 33],
                      [218, 108, 22]])/255

# exit()
MatValLfp_Todos=[]
Targets_corr_todos=[]
for i in np.arange(178,179,1):
    print('Archivo: ' +Dir[i])
    #base=12
    #i=12
    print('Vamos en la iteración--->'+str(i))
    time.sleep(2)
    mat = scipy.io.loadmat(path+'/'+Dir[i])
    (trial,spikes)=mat2Df(mat)
    
    event_names=trial.columns
    eventOfInterest=event_names[[12, 13, 14, 15, 16, 17, 22, 23, 24, 25, 26, 27, 30, 31]]
    alignMat_corr=[]
    alignMat_incorr=[]
    neurons_corr=[]
    neurons_incorr=[]
    Electrodes=[0]
 #   AngRot=[3.2, 1.6, .8, .4, .2, .1, -3.2, -1.6, -.8, -.4, -.2, -.1]
    
    
    for event in alignEvents:
        alineado,neurons=alinear(trial,spikes,event)
        #if event == alignEvents[0]:
        #    lfp=np.array(alineado['lfp'][:])
        #
        alignMat_corr.append(alineado.loc[alineado['correcto']==1])
        alignMat_incorr.append(alineado.loc[alineado['correcto']==0])
        neurons_corr.append(neurons.loc[alineado['correcto']==1])
        neurons_incorr.append(neurons.loc[alineado['correcto']==0])
    RasterPlot2(alignMat_corr,neurons_corr,Columns, eventOfInterest, colorMatrix, Dir[i][:-4])
    if len(np.unique(alignMat_corr[0]['anguloRotacion']))<12:
        continue
    else:
        # targets_corr=Targets(alignMat_corr[2],problems)
        for Electrode in Electrodes:
            if not alignMat_corr[0]['lfp'][alignMat_corr[0].index[0]].shape[0]>Electrode or NonBrokenElectrodes[Dir[i][:-4]][Electrode] == 0:
                continue
            else:
                (MatValLfp, angRot)= Val (alignMat_corr[2],Electrode,filterCoeff,False,True)
                # (TrimedMat, TrimedAngRot) = Trim (MatValLfp,angRot,100)
                targets_corr=Targets2(np.array(angRot),problems)
                # targets_corr=Targets(alignMat_corr[2],problems)
                # plotRastLFP(alignMat_corr,Columns,colorMatrix,eventOfInterest,filterCoeff,Dir[i][:-4],Electrode)
                # Energy,X=energy(alignMat_corr[2],winsize,Electrode,lev,fs,filterCoeff)
                # MFCCs_Mat=np.array(Mfcc(alignMat_corr[2],fs,Electrode,emphasize,
                #                          frame_length,frame_step,hamming,nfilt,NFFT,
                #                          num_ceps,sig_lift,filterCoeff))
                
                # (Cwtmatr,freq,X,widths)=CWTAverage(np.array(MatValLfp),targets_corr,fs,
                #                             'morl',targets_corr,problems,10,100,8)
                
                # pltCwt(Cwtmatr,X,targets_corr,problems,freq,Dir[i],Electrode)
                
                # (Cwtmatr2,freq,X2)=CWT(alignMat_corr[2],Electrode,fs,'morl',
                                     # filterCoeff)
                #MatValLfp_Todos.append(MatValLfp)
                # Targets_corr_todos.append(targets_corr) 
            # if Electrode==0 and i==base:
                # MatValLfp_Todos=MatValLfp.copy()
                # CWT_corr_Todos=Cwtmatr
                # MFCCs_corr_Todos=MFCCs_Mat
                # Targets_corr_todos=targets_corr.copy()
                # Eventos_Totales=[len(alignMat_corr[0].index)]
                # Eventos_Totales=[len(angRot)]
            # else:
                # MatValLfp_Todos=np.concatenate((MatValLfp_Todos,MatValLfp),axis=0)
                # MFCCs_corr_Todos=np.concatenate((MFCCs_corr_Todos,MFCCs_Mat),axis=0)
                # CWT_corr_Todos=np.concatenate((CWT_corr_Todos,Cwtmatr),axis=0)
                # Targets_corr_todos[problems[0]]=np.concatenate((Targets_corr_todos[problems[0]],
                                                            # targets_corr[problems[0]]))
                # Targets_corr_todos[problems[1]]=np.concatenate((Targets_corr_todos[problems[1]],
                                                            # targets_corr[problems[1]]))
                # Targets_corr_todos[problems[2]]=np.concatenate((Targets_corr_todos[problems[2]],
                                                            # targets_corr[problems[2]]))
                # Eventos_Totales.append(len(alignMat_corr[0].index))
                # Eventos_Totales.append(len(angRot))
# Estadisticos (MatValLfp, angRot, 50)
# Estadisticos2 (MatValLfp, angRot, 10,500,1)
# Estadisticos3 (TrimedMat, TrimedAngRot, 100, 1)
# Estadisticos3 (MatValLfp, angRot, 100, 1)
# TransHilbert(MatValLfp, angRot)          
# CNN(MFCCs_corr_Todos,Targets_corr_todos[problems[0]],winsize)
# X=np.array(X)               
# uno=sum(MFCCs_Mat[targets_corr[problems[1]]==0][:][:])
# dos=sum(MFCCs_Mat[targets_corr[problems[1]]==1][:][:])
# tres=sum(MFCCs_Mat[targets_corr[problems[1]]==2][:][:])
# quat=sum(MFCCs_Mat[targets_corr[problems[1]]==3][:][:])
# cinco=sum(MFCCs_Mat[targets_corr[problems[1]]==4][:][:])
# seis=sum(MFCCs_Mat[targets_corr[problems[1]]==5][:][:])

# uno_lfp=sum(X[targets_corr[problems[1]]==0])
# dos_lfp=sum(X[targets_corr[problems[1]]==1])
# tres_lfp=sum(X[targets_corr[problems[1]]==2])
# quat_lfp=sum(X[targets_corr[problems[1]]==3])
# cinco_lfp=sum(X[targets_corr[problems[1]]==4])
# seis_lfp=sum(X[targets_corr[problems[1]]==5])


# fig, axs=plt.subplots(7,1,figsize=(20,8))
# plt.savefig('/home/isaac/Documents/Doctorado CIC/Victor/RasterPlot/'+Title+key, 
#                 facecolor='w', edgecolor='w')
#             plt.close()

# axs[0].imshow(uno[1:,3:],aspect='auto',cmap='jet')
# axs[1].imshow(dos[1:,3:],aspect='auto',cmap='jet')
# axs[2].imshow(tres[1:,3:],aspect='auto',cmap='jet')
# axs[3].imshow(quat[1:,3:],aspect='auto',cmap='jet')
# axs[4].imshow(cinco[1:,3:],aspect='auto',cmap='jet')
# axs[5].imshow(seis[1:,3:],aspect='auto',cmap='jet')
# axs[6].plot(uno_lfp,color=[1,0,0],alpha=1)
# axs[6].plot(dos_lfp,color=[1,.1,0],alpha=1)
# axs[6].plot(tres_lfp,color=[1,.2,0],alpha=1)
# axs[6].plot(quat_lfp,color=[1,.3,0],alpha=1)
# axs[6].plot(cinco_lfp,color=[1,.4,0],alpha=1)
# axs[6].plot(seis_lfp,color=[1,.5,0],alpha=1)
    