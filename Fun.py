#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 20:30:17 2021

@author: isaac
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
from scipy import signal
from getFeatures import getFeatures as gF
from MFCCs import MFCCs 
from scipy.fftpack import dct
import pywt as pw
from scipy.signal import hilbert
from matplotlib.animation import FuncAnimation
from scipy.signal import find_peaks

# Gradientes de color =========================================================
def hex_to_RGB(hex):
  ''' "#FFFFFF" -> [255,255,255] '''
  # Pass 16 to the integer function for change of base
  return [int(hex[i:i+2], 16) for i in range(1,6,2)]

def RGB_to_hex(RGB):
  ''' [255,255,255] -> "#FFFFFF" '''
  # Components need to be integers for hex to make sense
  RGB = [int(x) for x in RGB]
  return "#"+"".join(["0{0:x}".format(v) if v < 16 else
            "{0:x}".format(v) for v in RGB])

def color_dict(gradient):
  ''' Takes in a list of RGB sub-lists and returns dictionary of
    colors in RGB and hex form for use in a graphing function
    defined later on '''
  return {"hex":[RGB_to_hex(RGB) for RGB in gradient],
      "r":[RGB[0] for RGB in gradient],
      "g":[RGB[1] for RGB in gradient],
      "b":[RGB[2] for RGB in gradient]}

def linear_gradient(start_hex, finish_hex, n):
  ''' returns a gradient list of (n) colors between
    two hex colors. start_hex and finish_hex
    should be the full six-digit color string,
    inlcuding the number sign ("#FFFFFF") '''
  # Starting and ending colors in RGB form
  s = hex_to_RGB(start_hex)
  f = hex_to_RGB(finish_hex)
  # Initilize a list of the output colors with the starting color
  RGB_list = [s]
  # Calcuate a color at each evenly spaced value of t from 1 to n
  for t in range(1, n):
    # Interpolate RGB vector for color at the current value of t
    curr_vector = [
      int(s[j] + (float(t)/(n-1))*(f[j]-s[j]))
      for j in range(3)
    ]
    # Add it to our list of output colors
    RGB_list.append(curr_vector)

  return color_dict(RGB_list)


# =============================================================================
def mat2Df(mat):
    
    # event_names2=['anguloInicio','anguloRotacion','velocidad','tiempo',
    #         'tiempoMedido','categoria','anguloTarg','respuesta','correcto',
    #         'digitalInfo','timeStamp','waitCueIni','manosFijasIni','cmdIni',
    #         'waitCueFin','touchCueIni','manosFijasFin','touchIni','cmdStim',
    #         'movIni','movFin','stimFin','touchCueFin','touchFin',
    #         'waitRespIni','targOn','waitRespFin','targOff','robSignal',
    #         'robTimeSec','robMovIni','robMovFin','lfp','lfpTime']
    
    trial=mat['e']['trial'][0][0][0][:] #las primeras 3 dimensiones solo
    spikes=mat['e']['spikes'][0][0][0][:]
    
    event_names=trial.dtype.descr
    spike_names=spikes.dtype.descr
    for spknames in range(len(spike_names)):
        spike_names[spknames]=spike_names[spknames][0]
        
    for evntnames in range(len(event_names)):
        event_names[evntnames]=event_names[evntnames][0]
    
    #tienen 1 valor, por lo que sobre la 4ta se debe de iterar
    #y la 5ta es el nombre de la columna

    for event in event_names: 
        for i in range(len(trial)):
            if trial[i][event].shape[0] == 1 or trial[i][event].shape[1] == 1:
                trial[i][event]=trial[i][event].flatten()
            
    for neuron in spike_names: 
        for i in range(len(spikes)):
            if spikes[i][neuron].shape[0] == 1 or spikes[i][neuron].shape[1] == 1:
                spikes[i][neuron]=spikes[i][neuron].flatten()
            
    
    SpikesDF=pd.DataFrame(spikes[:][spike_names[:]])
    TrialDF=pd.DataFrame(trial[:][event_names[:]])
    return TrialDF,SpikesDF
    

# =============================================================================
def alinear(trial,spikes,alignEvent):
    
    events=trial.copy()
    neurons=spikes.copy()
    
    for n in events.index:
        if not events['robMovIni'][n] > 0:
            events.drop(index=n,inplace=True)
            neurons.drop(index=n,inplace=True)
    
    for n in events.index:
        events['anguloRotacion'][n]=np.round(events['anguloRotacion'][n]*10)/10
    
    for n in events.index:
        alignTime = events[alignEvent][n]
        #print(n)
        events['cmdIni'][n]            = events['cmdIni'][n] - alignTime ;
        events['manosFijasIni'][n]     = events['manosFijasIni'][n] - alignTime;
        events['manosFijasFin'][n]     = events['manosFijasFin'][n]  - alignTime;
        events['waitCueIni'][n]        = events['waitCueIni'][n] - alignTime;
        events['waitCueFin'][n]        = events['waitCueFin'][n] - alignTime;
        events['touchIni'][n]          = events['touchIni'][n]  - alignTime;
        events['touchFin'][n]          = events['touchFin'][n]  - alignTime;
        events['touchCueIni'][n]       = events['touchCueIni'][n] - alignTime;
        events['touchCueFin'][n]       = events['touchCueFin'][n] - alignTime;
        events['waitRespIni'][n]       = events['waitRespIni'][n] - alignTime;
        events['waitRespFin'][n]       = events['waitRespFin'][n] - alignTime;
        events['targOn'][n]            = events['targOn'][n] - alignTime;
        events['targOff'][n]           = events['targOff'][n] - alignTime;
        events['cmdStim'][n]           = events['cmdStim'][n] - alignTime;
        events['movIni'][n]            = events['movIni'][n] - alignTime;
        events['stimFin'][n]           = events['stimFin'][n] - alignTime;
        events['movFin'][n]            = events['movFin'][n] - alignTime;
        events['robTimeSec'][n]        = events['robTimeSec'][n] - alignTime;
        events['lfpTime'][n]           = events['lfpTime'][n] - alignTime;
        #events[digitalInfo[:,1]][n]  = events(n).digitalInfo(:,1) - alignTime;
        events['robMovIni'][n]         = events['robMovIni'][n] - alignTime;
        events['robMovFin'][n]         = events['robMovFin'][n]  - alignTime;
        
        for nname in neurons.keys():
            neurons[nname][n]= neurons[nname][n]- alignTime
    
    
    neurons['anguloRotacion']=events['anguloRotacion']
    events.sort_values(by=['anguloRotacion'],inplace=True, ascending=False,ignore_index=True)
    izq=events[events['anguloRotacion']>0]
    der=events[events['anguloRotacion']<0]
    der.sort_values(by=['anguloRotacion'],inplace=True, ascending=True,ignore_index=True)
    Events=pd.concat([izq,der],ignore_index=True)
    neurons.sort_values(by=['anguloRotacion'],inplace=True, ascending=False,ignore_index=True)
    izq=neurons[neurons['anguloRotacion']>0]
    der=neurons[neurons['anguloRotacion']<0]
    der.sort_values(by=['anguloRotacion'],inplace=True, ascending=True,ignore_index=True)
    Neurons=pd.concat([izq,der],ignore_index=True)
    #neurons.drop('anguloRotacion',inplace=True,axis=1)
            
        
    return Events,Neurons
        
# =============================================================================
def plotRastLFP(alignMat,columns,colorMatrix,eventOfInterest,filterCoeff,Title,Electrode):
    pathRast='/home/isaac/Documents/Doctorado CIC/Victor/LFPFigures/RasterPlot'
    Top=[4.5, 1.4, 2.5, 1]
    fig=plt.figure(figsize=(20,9))
    Max=0
    [aF,bF,a60,b60]=filterCoeff
    for h in range(len(alignMat)):
        lfp=alignMat[h]['lfp'][:]
        for i in  alignMat[h].index:
            for j in columns[h]:
                if alignMat [h][eventOfInterest[j]][i]<=Top[h]:
                    plt.scatter(alignMat[h][eventOfInterest[j]][i]+Max,i,
                                color=colorMatrix[j],linewidths=.1)
                
            if Top[h]>max(alignMat[h]['lfpTime'][i]):
                continue
            else:
                pos=np.argmin(abs(alignMat[h]['lfpTime'][i]))    
                if h==0:
                    y=lfp[i][Electrode][pos:pos+round(Top[h]*1000)]
                    x=alignMat[h]['lfpTime'][i][np.arange(pos,pos+round(Top[h]*1000),10)]+Max
                else:
                    y=lfp[i][Electrode][pos-1000:pos+round(Top[h]*1000)]
                    x=alignMat[h]['lfpTime'][i][np.arange(pos-1000,pos+round(Top[h]*1000),10)]+Max
                
                sig= signal.filtfilt(bF,aF,y);    
                sig= signal.filtfilt(b60,a60,sig);
                Signal=(y/1200)+i;
                plt.plot(x,Signal[np.arange(0,len(Signal),10)],color=np.array([100,100,100])/255,linewidth=1)
            
                        
        Max=Max+Top[h]+1.5;          
    plt.xlabel('Time (s)')
    plt.ylabel('Number of trials')
    plt.axis([0, 13.9, -3, i+4])
    plt.suptitle(Title+', Electrode #'+str(Electrode))
    # plt.savefig(pathRast+'/'+Title+'Electrode'+str(Electrode))
    # plt.close('all')
    plt.show()

# =============================================================================
def energy(alignMat,winsize,Electrode,lev,fs,filterCoeff):
    # lfpDer_corr=alignMat[2][['lfp','lfpTime']][alignMat_corr[2]['anguloRotacion']<0]      
    # lfpIzq_corr=alignMat[2][['lfp','lfpTime']][alignMat_corr[2]['anguloRotacion']>0]  
    [aF,bF,a60,b60]=filterCoeff
    #index_Der=alignMat[alignMat['anguloRotacion']<0].index
    #index_Izq=alignMat[alignMat['anguloRotacion']>0].index
    Top=2560
    numWin = int((Top+1024)/winsize)
    rshape=numWin*(lev+1)
    for i in alignMat.index:
        lfp=alignMat['lfp'][i][Electrode]
        sig= signal.filtfilt(bF,aF,lfp);    
        sig= signal.filtfilt(b60,a60,sig);
        time=alignMat['lfpTime'][i]
        pos=np.argmin(abs(time))
        x=sig[pos-1024:pos+Top]
        [ener, _, _, _]=gF(x,fs,winsize,lev) 
        if i == alignMat.index[0]:
            Energy=np.reshape(ener,(rshape,1))
            X=[x]
        else:
            Energy=np.concatenate((Energy,np.reshape(ener,(rshape,1))),axis=1)
            X.append(x)
    Energy=Energy.T
    
    
    # for i in index_Izq:
    #     lfp=alignMat['lfp'][i][Electrode]
    #     sig= signal.filtfilt(bF,aF,lfp);    
    #     sig= signal.filtfilt(b60,a60,sig);
    #     time=alignMat['lfpTime'][i]
    #     pos=np.argmin(abs(time))
    #     x=sig[pos-1024:pos+Top]
    #     [ener, _, _, _]=gF(x,fs,winsize,lev) 
    #     if i == index_Izq[0]:
    #         Energy_Izq=np.reshape(ener,(rshape,1))
    #     else:
    #         Energy_Izq=np.concatenate((Energy_Izq,np.reshape(ener,(rshape,1))),axis=1)
    # Energy_Izq=Energy_Izq.T
    # Energy=np.concatenate((Energy_Izq,Energy_Der))
    
    #target=np.concatenate((np.zeros(len(index_Izq)),np.ones(len(index_Der))))
    
    return Energy, X
         
# =============================================================================
def Mfcc(alignMat,fs,Electrode,emphasize,frame_length,frame_step,hamming,nfilt,NFFT,num_ceps,sig_lift,filterCoeff):
    
    [aF,bF,a60,b60]=filterCoeff
    #index_Der=alignMat[alignMat['anguloRotacion']<0].index
    #index_Izq=alignMat[alignMat['anguloRotacion']>0].index
    Top=2560
    #print(NFFT)
    for i in alignMat.index:
        lfp=alignMat['lfp'][i][Electrode]
        sig= signal.filtfilt(bF,aF,lfp);    
        sig= signal.filtfilt(b60,a60,sig);
        time=alignMat['lfpTime'][i]
        pos=np.argmin(abs(time))
        sig=sig[pos-1024:pos+Top]
        mfcc=MFCCs(sig,fs,emphasize,frame_length,frame_step,hamming,nfilt,NFFT,num_ceps,sig_lift)
        mfcc += np.abs(np.min(mfcc))
        
        if i == alignMat.index[0]:
            Mat=[mfcc]
        else:
            Mat.append(mfcc)
            
    # for i in index_Izq:
    #     lfp=alignMat['lfp'][i][Electrode]
    #     sig= signal.filtfilt(bF,aF,lfp);    
    #     sig= signal.filtfilt(b60,a60,sig);
    #     time=alignMat['lfpTime'][i]
    #     pos=np.argmin(abs(time))
    #     x=sig[pos-1024:pos+Top]
    #     mfcc=MFCCs(x, fs, emphasize, frame_length, frame_step, hamming, nfilt, 
    #                NFFT, num_ceps, sig_lift)
    #     mfcc += np.abs(np.min(mfcc))
    #     if i == index_Izq[0]:
    #         Mat_Izq=[mfcc]
    #     else:
    #         Mat_Izq.append(mfcc)
        
    return Mat

# =============================================================================
def Targets(alignMat,problems):
    targets={}
    aM=alignMat
    p1=np.flip(np.unique(aM['anguloRotacion']))
    targets[problems[0]]=np.concatenate((np.zeros(len(aM[aM['anguloRotacion']>0].index)),
                                   np.ones(len(aM[aM['anguloRotacion']<0].index))))
    targets[problems[1]]=np.concatenate((np.ones(len(aM[aM['anguloRotacion']==p1[0][0]].index))*5,
                                   np.ones(len(aM[aM['anguloRotacion']==p1[1][0]].index))*4,
                                   np.ones(len(aM[aM['anguloRotacion']==p1[2][0]].index))*3,
                                   np.ones(len(aM[aM['anguloRotacion']==p1[3][0]].index))*2,
                                   np.ones(len(aM[aM['anguloRotacion']==p1[4][0]].index))*1,
                                   np.ones(len(aM[aM['anguloRotacion']==p1[5][0]].index))*0,
                                   np.ones(len(aM[aM['anguloRotacion']==p1[6][0]].index))*0,
                                   np.ones(len(aM[aM['anguloRotacion']==p1[7][0]].index))*1,
                                   np.ones(len(aM[aM['anguloRotacion']==p1[8][0]].index))*2,
                                   np.ones(len(aM[aM['anguloRotacion']==p1[9][0]].index))*3,
                                   np.ones(len(aM[aM['anguloRotacion']==p1[10][0]].index))*4,
                                   np.ones(len(aM[aM['anguloRotacion']==p1[11][0]].index))*5))
    
    targets[problems[2]]=np.concatenate((np.zeros(len(aM[aM['anguloRotacion']==p1[0][0]].index)),
                                   np.ones(len(aM[aM['anguloRotacion']==p1[1][0]].index)),
                                   np.ones(len(aM[aM['anguloRotacion']==p1[2][0]].index))*2,
                                   np.ones(len(aM[aM['anguloRotacion']==p1[3][0]].index))*3,
                                   np.ones(len(aM[aM['anguloRotacion']==p1[4][0]].index))*4,
                                   np.ones(len(aM[aM['anguloRotacion']==p1[5][0]].index))*5,
                                   np.ones(len(aM[aM['anguloRotacion']==p1[6][0]].index))*6,
                                   np.ones(len(aM[aM['anguloRotacion']==p1[7][0]].index))*7,
                                   np.ones(len(aM[aM['anguloRotacion']==p1[8][0]].index))*8,
                                   np.ones(len(aM[aM['anguloRotacion']==p1[9][0]].index))*9,
                                   np.ones(len(aM[aM['anguloRotacion']==p1[10][0]].index))*10,
                                   np.ones(len(aM[aM['anguloRotacion']==p1[11][0]].index))*11))
    return targets

# =============================================================================
def Targets2(angRot,problems):
    targets={}
    # aM=alignMat_corr[2]
    # p1=np.flip(np.unique(aM[2]['anguloRotacion']))
    p1=np.flip(np.unique(angRot))
    targets[problems[0]]=np.concatenate((np.zeros(sum(angRot>0)),
                                   np.ones(sum(angRot<0))))
    targets[problems[1]]=np.concatenate((np.ones(sum(angRot==p1[0]))*5,
                                   np.ones(sum(angRot==p1[1]))*4,
                                   np.ones(sum(angRot==p1[2]))*3,
                                   np.ones(sum(angRot==p1[3]))*2,
                                   np.ones(sum(angRot==p1[4]))*1,
                                   np.ones(sum(angRot==p1[5]))*0,
                                   np.ones(sum(angRot==p1[6]))*5,
                                   np.ones(sum(angRot==p1[7]))*4,
                                   np.ones(sum(angRot==p1[8]))*3,
                                   np.ones(sum(angRot==p1[9]))*2,
                                   np.ones(sum(angRot==p1[10]))*1,
                                   np.ones(sum(angRot==p1[11]))*0))
    
    targets[problems[2]]=np.concatenate((np.zeros(sum(angRot==p1[0])),
                                   np.ones(sum(angRot==p1[1])),
                                   np.ones(sum(angRot==p1[2]))*2,
                                   np.ones(sum(angRot==p1[3]))*3,
                                   np.ones(sum(angRot==p1[4]))*4,
                                   np.ones(sum(angRot==p1[5]))*5,
                                   np.ones(sum(angRot==p1[6]))*6,
                                   np.ones(sum(angRot==p1[7]))*7,
                                   np.ones(sum(angRot==p1[8]))*8,
                                   np.ones(sum(angRot==p1[9]))*9,
                                   np.ones(sum(angRot==p1[10]))*10,
                                   np.ones(sum(angRot==p1[11]))*11))
    return targets

# =============================================================================
def CWT(alignMat,Electrode,fs,motherWavelet,filterCoeff):
    #widths = np.arange(0.1,6000,100)
    [aF,bF,a60,b60]=filterCoeff
    Top=2560
    #widths = np.logspace(np.log10(40),np.log10(1000),num=512)
    widths = []
    v=np.arange(3,10,1)
    M=16
    for J in v:   #generate the scales
            a1=[]
            for m in np.arange(1,M): 
                a1.append(2**(J+(m/M)))
            
            widths.append(a1)
    widths=np.array(widths)
    widths=widths.reshape(widths.shape[1]*widths.shape[0],)
    # widths=np.arange(40,80,1/8)# widths = []
    sp=1/fs
    X=[]
    Cwtmatr=[]
    for i in alignMat.index:
        print(i)
        lfp=alignMat['lfp'][i][Electrode]
        sig= signal.filtfilt(bF,aF,lfp);    
        sig= signal.hilbert(signal.filtfilt(b60,a60,sig))
        time=alignMat['lfpTime'][i]
        pos=np.argmin(abs(time))
        sig=sig[pos-1024:pos+Top]
        time=time[pos-1024:pos+Top]
        cwtmatr, freqs = pw.cwt(sig, widths, motherWavelet, sampling_period=sp)
        cwtmatr_real=abs(cwtmatr)
        for rows in range(cwtmatr.shape[0]):
            r=np.copy(cwtmatr[rows])
            cwtmatr[rows]=r/np.mean(r)
        # var=np.var(cwtmatr,axis=1)
        #cwtmatr=abs(cwtmatr)
        #cwtmatr=((cwtmatr/mn[:, np.newaxis])+np.min(cwtmatr))/np.max(cwtmatr)
        # cwtmatr=(cwtmatr+np.min(cwtmatr))/np.max(cwtmatr)
        # cwtmatr=cwtmatr/np.max(cwtmatr)
        X.append(sig)
        Cwtmatr.append(cwtmatr)
            
    
    return Cwtmatr,freqs,X


# =============================================================================
def freq2scale(MotherWavelet,fs,fmax,fmin):
    cf=pw.central_frequency(MotherWavelet)
    Scale=[]
    freqs=[fmax,fmin]
    for fr in freqs:
        Scale.append(np.round(np.log((cf*fs)/fr))/np.log(2))
    return Scale

# =============================================================================
def CWTAverage(MatValLfp,targets_corr,fs,motherWavelet,filterCoeff,
               problems,fmin,fmax,voices):
    
    sc=freq2scale(motherWavelet,fs,fmax,fmin)
    widths=[]
    v=np.arange(sc[0],sc[1],1)
    M=voices
    for J in v:   #generate the scales
            a1=[]
            for m in np.arange(1,M+1): 
                a1.append(2**(J+(m/M)))
            
            widths.append(a1)
    widths=np.array(widths)
    widths=widths.reshape(widths.shape[1]*widths.shape[0],)
    # widths=np.arange(40,80,1/8)# widths = []
    sp=1/fs
    cont=0
    Cwtmatr=[]
    X=[]
    for i in range(MatValLfp.shape[0]):
        print(i)
        # time=alignMat['lfpTime'][i]
        # pos=np.argmin(abs(time))
        pos=1000
        # sig= signal.filtfilt(bF,aF,lfp);    
        sig= signal.hilbert(MatValLfp[i])
        # sig=sig[pos-1024:pos+Top+.5]
        cwtmatr, freqs = pw.cwt(sig, widths, motherWavelet, sampling_period=sp)
        cwtmatr=np.abs(cwtmatr)
        for rows in range(cwtmatr.shape[0]):
            r=np.copy(cwtmatr[rows])
            cwtmatr[rows]=r/np.mean(r)
        # var=np.var(cwtmatr,axis=1)
        #cwtmatr=abs(cwtmatr)
        #cwtmatr=((cwtmatr/mn[:, np.newaxis])+np.min(cwtmatr))/np.max(cwtmatr)
        # cwtmatr=(cwtmatr+np.min(cwtmatr))/np.max(cwtmatr)
        # cwtmatr=cwtmatr/np.max(cwtmatr)
        Cwtmatr.append(cwtmatr)
        X.append(sig)
    return Cwtmatr,freqs,X, widths


#==============================================================================
def myUnique(angRot):
    degrees=[]
    HowMany=[]
    for x in angRot:
        if x not in degrees:
            degrees.append(x)
    degrees=np.array(degrees)
    
    for deg in degrees:
        HowMany.append(sum(angRot==deg))
    
    return degrees,HowMany

# =============================================================================
def Val(alignMat,Electrode,filterCoeff,Val,Return2Origin): #Val de valanceado 
    (_, counts)=np.unique(alignMat['anguloRotacion'], return_counts=True)
    angRot=np.array(alignMat['anguloRotacion'])
    angulos,_=myUnique(angRot)
    [aF,bF,a60,b60]=filterCoeff
    MatValLfp=[] #Val de valanceada
    angRot=[]
    # MatValTime=[]
    for ang in angulos: 
        #print(ang)
        a=alignMat['lfp'][alignMat['anguloRotacion']==ang[0]]
        if Val == True:
            index=np.random.choice(a.index,size=min(counts),replace=False)
        else: 
            index=a.index
        # cont=0
        # plt.figure()
        for idx in index:
            sig=a[idx][Electrode]
            sig= signal.filtfilt(bF,aF,sig);    
            sig= signal.filtfilt(b60,a60,sig);
            if Return2Origin: #Return2Origin es una técnica para eliminar periodos
                Dist=np.zeros(len(sig)-1)
                for i in range(len(Dist)):
                    Dist[i]=sig[i+1]-sig[i]
                sig=Dist
            time=alignMat['lfpTime'][idx]
            zeroTime=np.argmin(abs(time))
            angR=alignMat['anguloRotacion'][idx]
            angRot.append(angR)
            MatValLfp.append(sig[zeroTime-1024:zeroTime+2560])
    #MatValLfp=np.array(MatValLfp)
    #angRot=np.array(angRot)
    
    return MatValLfp, angRot


# =============================================================================
def Trim (MatValLfp,angRot,Window):
    Mat=MatValLfp.copy()
    LenSig=len(Mat[0])
    TrimedMat=[]
    TrimedAngRot=[]
    for i in range(len(Mat)):
        stdMean=np.zeros(LenSig-Window)
        for j in range(LenSig-Window): 
                    stdMean[j]=np.std(Mat[i][0+j:Window+j])
        
        peaks, _ = find_peaks(stdMean, prominence=(np.mean(stdMean), None)) 
        peaks=peaks[(1000<peaks) & (peaks<1600)]
    
        if len(peaks):
            stdMax=max(stdMean[peaks])
            booleanVec=sum(abs(Mat[i][0:1600])>stdMax)
            if booleanVec <=100:
                print(i)
                TrimedMat.append(Mat[i])
                TrimedAngRot.append(angRot[i])
            
    return TrimedMat, TrimedAngRot 


# =============================================================================
def Estadisticos (MatValLfp, angRot, Window):
    fig, axs=plt.subplots(6,2,figsize=(20,8))
    # fig2, axs2=plt.subplots(2,1,figsize=(20,8))
    Mat=np.array(MatValLfp)
    fila=-1
    col=0
    MeanStdLine=np.arange(Window/2,(Mat.shape[1])-Window/2,1)
    for ang in np.flip(np.unique(angRot)):
        if ang == -0.1:    
            col=1
        elif ang < -0.1:
            fila-=1
        else:
            fila+=1
        index=[i for i, x in enumerate(angRot==ang) if x]
        axs[fila][col].plot(Mat[index].T, color=[.7,.7,.7])
        # axs[fila][col].plot(Mat[index[-1:]].T, color=[0,0,0])
        axs[fila][col].set_title(str(ang))
        
       
        mean=[]
        std=[]
        Sig=np.mean(Mat[index],axis=0)  
        for j in range(Mat.shape[1]-Window):
            mean.append(np.mean(Sig[0+j:Window+j]))
            std.append(np.std(Sig[0+j:Window+j]))
        mean=np.array(mean)
        std=np.array(std)
            
        # mean=np.mean(Mat[index],axis=0)
        # std=np.std(Mat[index],axis=0)
        axs[fila][col].plot(MeanStdLine,mean,color='r')
        axs[fila][col].plot(MeanStdLine,mean+std,color='b')
        axs[fila][col].plot(MeanStdLine,mean-std,color='b')
        axs[fila][col].fill_between(MeanStdLine,mean+std,mean-std,alpha=0.2)
        
        # if ang==3.2:
        #     axs2[0].plot(Mat[index].T, color=[.7,.7,.7])
        #     axs2[0].plot(Mat[index[-3:-2]].T, color=[0,0,0])
        #     axs2[0].plot(mean,color='r')
        #     axs2[0].plot(mean+std,color='b')
        #     axs2[0].plot(mean-std,color='b')
        #     axs2[0].set_title(str(ang))
        #     axs2[0].fill_between(np.arange(0,len(mean),1),mean+std,mean-std,alpha=0.2)
            
        # if ang==-3.2:
        #     axs2[1].plot(Mat[index].T, color=[.7,.7,.7])
        #     axs2[1].plot(Mat[index[-3:-2]].T, color=[0,0,0])
        #     axs2[1].plot(mean,color='r')
        #     axs2[1].plot(mean+std,color='b')
        #     axs2[1].plot(mean-std,color='b')
        #     axs2[1].set_title(str(ang))
        #     axs2[1].fill_between(np.arange(0,len(mean),1),mean+std,mean-std,alpha=0.2)
        # print(fila, ang)

        
# =============================================================================

def Estadisticos2 (MatValLfp, angRot, WinMin, WinMax, stride):
    # fig, axs=plt.subplots(6,2,figsize=(20,8))
    fig = plt.figure(figsize=(40,15))
    ax1 = fig.add_subplot(621)
    ax2= fig.add_subplot(623)
    ax3= fig.add_subplot(625)
    ax4= fig.add_subplot(627)
    ax5= fig.add_subplot(629)
    ax6= fig.add_subplot(6,2,11)
    ax7= fig.add_subplot(622)
    ax8= fig.add_subplot(624)
    ax9= fig.add_subplot(626)
    ax10= fig.add_subplot(628)
    ax11= fig.add_subplot(6,2,10)
    ax12= fig.add_subplot(6,2,12)
    Mat=np.array(MatValLfp)
    
        
    def animate(i):
        print('Entro')
        for j in np.arange(1,13,1):
            eval('ax'+str(j)+'.cla()')
            eval('ax'+str(j)+'.grid()')
            
        Window=i
        line=np.concatenate((np.zeros(200),np.ones(Window)*5000,[0]))
        cont=0
        MeanStdLine=np.arange(Window/2,(Mat.shape[1])-Window/2,1)
        var='izq'
        for ang in np.flip(np.unique(angRot)):
            if ang == -0.1:    
                cont=12
            elif ang < -0.1:
                cont-=1
            else:
                cont+=1
            index=[i for i, x in enumerate(angRot==ang) if x]

            
           
            mean=np.zeros((len(index),Mat.shape[1]-Window))
            for j in range(len(index)):
                for k in range(Mat.shape[1]-Window): 
                    mean[j,k]=np.mean(Mat[index[j]][0+k:Window+k])
            
            mean=np.array(mean)
            Mean=np.mean(mean,axis=0)
            std=np.std(mean,axis=0)
            eval('ax'+str(cont)+'.plot(line, color=[0,.5,0])')
            eval('ax'+str(cont)+'.axvline(x=1000,color=[.5,0,0])')
            eval('ax'+str(cont)+'.plot(MeanStdLine,mean.T, color=[.7,.7,.7])')
            eval('ax'+str(cont)+'.plot(MeanStdLine,Mean, color=[.811,.203,.462])')
            eval('ax'+str(cont)+'.set_title('+str(ang)+')')
            eval('ax'+str(cont)+'.set_ylim(-10000, 15000)')
            eval('ax'+str(cont)+'.set_xlim(0, 3560)')
            # eval('ax'+str(cont)+'.grid(b=True)')
        plt.suptitle('Window= '+str(Window))
        plt.pause(.01)
            
    ani = FuncAnimation(fig, func=animate, 
                        frames=WinMax,
                        interval=1,repeat=False)
    
    ani.save('test2.gif',fps=15,writer='PillowWriter')
    plt.show()

    
# =============================================================================
def Estadisticos3 (TrimedMat, TrimedAngRot, Window, stride):
    
    fig = plt.figure(figsize=(20,15))
    ax1 = fig.add_subplot(721)
    ax2= fig.add_subplot(723)
    ax3= fig.add_subplot(725)
    ax4= fig.add_subplot(727)
    ax5= fig.add_subplot(729)
    ax6= fig.add_subplot(7,2,11)
    ax7= fig.add_subplot(722)
    ax8= fig.add_subplot(724)
    ax9= fig.add_subplot(726)
    ax10= fig.add_subplot(728)
    ax11= fig.add_subplot(7,2,10)
    ax12= fig.add_subplot(7,2,12)
    ax13= fig.add_subplot(717)
    Mat=np.array(TrimedMat)
    angRot=np.array(TrimedAngRot)
    
    axisGauss=np.arange(0,Mat.shape[1],1) 
    
    colors=linear_gradient('#E84A5F','#2A363B',12)
    color=np.array([colors['r'],colors['g'],colors['b']]).T/255
    
    cont=0
    MeanStdLine=np.arange(Window/2,(Mat.shape[1])-Window/2,1)
    for ang in np.flip(np.unique(angRot)):
        if ang == -0.1:    
            cont=12
        elif ang < -0.1:
            cont-=1
        else:
            cont+=1
        index=[i for i, x in enumerate(angRot==ang) if x]
        Mean=np.zeros((len(index),Mat.shape[1]-Window))
        #Sig=np.mean(Mat[index],axis=0)
        for j in range(len(index)):
            for k in range(Mat.shape[1]-Window): 
                Mean[j,k]=np.mean(Mat[index[j]][0+k:Window+k])
        
        #mean=np.array(mean)
        
        # infls=[]
        # for sig in Std:
        #     peaks, _ = find_peaks(sig, prominence=(np.mean(sig), None)) 
        #     peaks=peaks[(1000<peaks) & (peaks<1700)]
        #     MaxPeak=np.argmax(sig[peaks])
        #     infls.append(peaks[MaxPeak]+Window/2)
            
        # infls=np.array(infls)
        MeanMean=np.mean(Mean,axis=0)
        # meanInfls=np.mean(infls)
        # stdInfls=np.std(infls)
        # Gauss=scipy.stats.norm(meanInfls, stdInfls)
        eval('ax'+str(cont)+'.axvline(x=1000,color=[.5,0,0])')
        #eval('ax'+str(cont)+'.axvline(x='+str(meanInfls)+',color=[.38,.54,.23])')
        #for infl in infls:
        #    eval('ax'+str(cont)+'.axvline(x='+str(infl)+',color=[.38,.54,.23],alpha=.3)')
        eval('ax'+str(cont)+'.plot(Mat[index].T, color=''[.7,.7,.7])')
        # eval('ax'+str(cont)+'.plot(Mean.T, color=''[.7,.7,.7])')
        eval('ax'+str(cont)+'.plot(MeanStdLine,MeanMean, color='+str([color[cont-1][0],
                                                                 color[cont-1][1],
                                                                 color[cont-1][2]])+')')
        eval('ax'+str(cont)+'.set_title('+str(ang)+')')
        #eval('ax'+str(cont)+'.set_ylim(-10000, 15000)')
        eval('ax'+str(cont)+'.set_xlim(0, 3560)')
        #eval('ax13.plot(axisGauss,Gauss.pdf(axisGauss), color='+str([color[cont-1][0],
        #                                                         color[cont-1][1],
        #                                                         color[cont-1][2]])+')')
        # eval('ax'+str(cont)+'.grid(b=True)')
    plt.suptitle('Window= '+str(Window))
    

# =============================================================================

def TransHilbert(MatValLfp, angRot):
    fig, axs=plt.subplots(2,6,figsize=(20,8),subplot_kw={'projection': 'polar'})
    fig2, axs2=plt.subplots(figsize=(20,8),subplot_kw={'projection': 'polar'})
    Mat=np.array(MatValLfp)
    fila=0
    
    for ang in np.flip(np.unique(angRot)):
        index=[i for i, x in enumerate(angRot==ang) if x]
        
        if ang>0:
            col=0
        if ang == -0.1: 
            col=1
            fila-=6
        
        Amp=[]
        Phase=[]
        for j in index:
            z= hilbert(Mat[j]) #form the analytical signal
            inst_amplitude = np.abs(z) #envelope extraction
            inst_phase = np.unwrap(np.angle(z))#inst phase
            axs[col][fila].scatter(inst_phase, inst_amplitude, color=[.7,.7,.7], 
                                   cmap='hsv', alpha=0.75)
            
            Amp.append(inst_amplitude)
            Phase.append(inst_phase)
            if col==0 and fila==0:
                axs2.scatter(inst_phase, inst_amplitude, color=[.7,.7,.7], 
                                   cmap='hsv', alpha=0.75)
        
        AmpMean=np.mean(np.array(Amp),axis=0)
        PhaseMean=np.mean(np.array(Phase),axis=0)
        axs[col][fila].scatter(PhaseMean, AmpMean, c=PhaseMean, 
                                   cmap='hsv')
        axs[col][fila].set_ylim(0,20000)
        
        if col==0 and fila==0:
                axs2.scatter(PhaseMean, AmpMean, c=PhaseMean, 
                                   cmap='hsv')
                axs2.set_ylim(0,20000)
        fila+=1


#==============================================================================

def getArgumentValue(argument,defaultValue,args):
    
    
    for k in range(len(args)):
       if isinstance(args[k], str):
          if argument.lower() == args[k].lower():
             break
    print(k)     
       
    if k!=len(args)-1:
        value = args[k+1]; # Return the value, following the ARGUMENT string.
    else:
        value = defaultValue;
    
    return value


#==============================================================================

def firingrate(SpikeTimes, TimeSamples, *args):
    
# Calculate the firing rate at each TimeSample
#
# FRATES = firingrate(SPIKETIMES, TIMESAMPLES, OPTIONS)
# Examples:
# fRates = firingrate(SpikeTimes, TimeSamples, 'FilterType', 'exponential',...
#                    'TimeConstant', 0.05);
# fRates = firingrate(SpikeTimes, TimeSamples, 'FilterType', 'boxcar',...
#                    'TimeConstant', 0.05);
# fRates = firingrate(SpikeTimes, TimeSamples, 'Attrit',      [0 10]);
# 
# SpikeTimes can be either a vector of spike times or a cell array 
# containing one vector per cell.
#
# Time unit: seconds
# fRates: spikes/second
#
# Options: 
# 'FilterType'   ,  'boxcar' or 'exponential'
# 'TimeConstant' ,  [.05] If 'Filtertype' is 'exponential' TimeConstant sets the
#                         decay rate of the exponential (see equation below).
#                         If 'Filtertype' is 'boxcar' TimeConstant sets
#                         the total length of the window.
# 'Attrit'       ,  [tIni tEnd] Either only one pair, or the number of rows
#                               must equal the number of spike vectors. 
#                               Attrit will put NANs at every TimeSample 
#                               falling outside the tIni tEnd bounds.
# 'Normalize'    ,  [TimeConstant Center] Always uses a boxcar
#

    if isinstance(SpikeTimes, pd.DataFrame)==False:
        SpikeTimes=pd.DataFrame(SpikeTimes)
        SpikeTimes.reset_index(inplace=True,drop=True)
    
    key=SpikeTimes.keys()[0]
    FilterType = getArgumentValue('FilterType','boxcar', args)
    # Time constant in seconds
    TimeConstant = getArgumentValue('TimeConstant',0.05, args) 
    # Attrition times
    attrit = np.array(getArgumentValue('attrit',[TimeSamples[0], TimeSamples[-1]],args))
    # Normalization width and center
    normWidthCent = getArgumentValue('Normalize',[], args) 
    
    #if attrit.ndim >1: 
    
    # Get the number of trials
    ntrials = len(SpikeTimes)

    # Initialize fRates matrix (a row for each trial)
    fRates = np.empty((ntrials,len(TimeSamples)))  
    fRates[:] = np.nan       
    
    for indx in SpikeTimes.index:
        if attrit.ndim>1: #If only more then one pair or attrition times were provided use a different pair for each trial. 
           subtimesamp = [(TimeSamples<= attrit[indx,1] & TimeSamples>= attrit[indx,0])];
        else: #If only one pair or attrition times were provided use that pair for all the trials.
           subtimesamp = [(TimeSamples<= attrit[1])&(TimeSamples>= attrit[0])]
        
        timeSamples = TimeSamples[subtimesamp]
        spkT  = SpikeTimes[key][indx]                 # Get the trial spikes.
        spkT  = spkT.reshape(len(spkT),1)
        spkT  = spkT*np.ones((1,len(timeSamples)))
        timesamples = np.ones((1,spkT.shape[0])).T *timeSamples
        
        
        if FilterType == 'boxcar': #Counts spikes between -TimeConstant/2 and +TimeConstant2 of each time sample (i.e. it extends into the past and future) 
            fRate  = (np.sum([(spkT>=(timesamples-TimeConstant/2))&(spkT<(timesamples+TimeConstant/2))],axis=1)/TimeConstant)
        elif FilterType == 'exponential':
            
            SelectSpikesUpTo = TimeConstant * 7.5; # Time window within which spike will contribute to the firing rate of each time sample.
            SpikesBeforeTimeSample=np.array([(spkT>(timesamples-SelectSpikesUpTo)) & (spkT<=timesamples)]).reshape(spkT.shape[0],spkT.shape[1])
            DistanceToTimeSample = spkT - timesamples;
            DistanceToTimeSample = abs(np.multiply(SpikesBeforeTimeSample,DistanceToTimeSample));
            fRate = np.multiply(1/TimeConstant ,np.exp(-DistanceToTimeSample/TimeConstant));
            fRate[SpikesBeforeTimeSample==False]=0; # Remove those distances that we are not considering.
            fRate = np.sum(fRate,axis=0);
        else:
            raise Exception(['FilterType '+ FilterType +' not supported'])
            
        fRates[indx] = fRate
    
    return fRates

# =============================================================================

def RasterPlot(alignMat,neurons_corr,Columns, eventOfInterest, colorMatrix, Title):
# Necesitas valancear las clases ... pruebalo 
# alignMat=alignMat_corr
#colors=linear_gradient('#34ebe5','#344ceb',,6)
#colors=linear_gradient('#eb4034','#ebdf34',6)
    # colors=linear_gradient('#ebdf34','#eb4034',6)
    # colorizq=np.array([colors['r'],colors['g'],colors['b']]).T/255
    # colors=linear_gradient('#34ebe5','#344ceb',6)
    # colorder=np.array([colors['r'],colors['g'],colors['b']]).T/255
    # colors=np.concatenate([colorizq,colorder])
    colors=['r','b']
    Top=[4.5, 1.4, 2.5, 1]
    Start=[0, -1, -1, -1]
    samples=.1
    
    
    
    neuronsKeys=neurons_corr[0].keys()[:-1]
    plt.close('all')
    # angRot=np.array(neurons_corr[2]['anguloRotacion'])
    angRot=neurons_corr[2]['anguloRotacion']
    angDer=np.array(angRot[angRot<0].index)
    angIzq=np.array(angRot[angRot>0].index)
    angulos=[angIzq,angDer]
    HowMany=[len(angIzq),len(angDer)]
    #angulos,HowMany=myUnique(angRot)
    #Nota: agragar corr-incorr
    
    length_checker = np.vectorize(len)
    
    fRates=[]
    
    for key in neuronsKeys:
        if key[-3:] != '255' and key[-1:] != '0':
            fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2,1]}, figsize=(20,8))
            plt.suptitle(Title+key)
            Slide=0
            for h in range(len(neurons_corr)):
                TimeSamples=np.arange(Start[h],Top[h]+samples,samples)
                
            #for h in range(1):
                #h=3
                cont=0
                # for ang in angulos:
                for ang in angulos:
                    compactTrail=np.empty(1,)
                    trial2plot=neurons_corr[h][key]
                    # trial2plot=trial2plot[neurons_corr[h]['anguloRotacion']==ang[0]]
                    trial2plot=trial2plot[ang]
                    #trail2plot=trail2plot.reset_index()
                    offsets = np.arange(0,len(trial2plot),1)+sum(HowMany[:cont])
                    # offsets = np.arange(0,len(trial2plot),1)+len(ang)

                    #offsets = np.array(trail2plot.index)
                    slicedTrails=[]
                    for trial in trial2plot:
                        sliced=trial[(trial>=Start[h])&(trial<=Top[h])]+Slide
                        slicedTrails.append(sliced)
                        compactTrail=np.concatenate([compactTrail,sliced])
                    # axs[0].eventplot(slicedTrails,colors=colors[cont],lineoffsets=offsets)
                    if offsets !=[]:
                        axs[0].eventplot(slicedTrails,color=colors[cont],lineoffsets=offsets)

                    #eval('fRate'+str(cont)+"=firingrate(trial2plot,TimeSamples,'FilterType','exponential','TimeConstant', 0.5)")
                    #fRate=firingrate(trial2plot,TimeSamples,'FilterType','exponential','TimeConstant', 0.5)
                    #axs[1].plot(TimeSamples+Slide,np.mean(fRate,axis=0),color=colors[cont])
                    #plt.eventplot(neuron2plot)
                    # sns.kdeplot(compactTrail,
                    #               ax=axs[1],
                    #               color=RGB_to_hex(colors[cont]*255))
                    
                    if compactTrail!=[]:
                        sns.histplot(compactTrail, 
                                   ax=axs[1],color=colors[cont],
                                   fill=False, kde=True, binwidth=.05, common_norm=False)
                    cont+=1
                for j in Columns[h]:
                    axs[0].eventplot(alignMat[h][eventOfInterest[j]]+Slide,colors=colorMatrix[j],linewidths=3)
                                    
             
                        
                axs[1].axvline(x=0+Slide,color=colorMatrix[Columns[h][0]])
                
                
                Slide+=Top[h]+1.5
            plt.show()
            # plt.savefig('/home/isaac/Documents/Doctorado CIC/Victor/RasterPlot/'+Title+key, 
            #     facecolor='w', edgecolor='w')
            # plt.close()