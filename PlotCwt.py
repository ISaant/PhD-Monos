#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 19:25:47 2021

@author: isaac
"""

import numpy as np
import matplotlib.pyplot as plt

def PlotCWT (Cwtmatr,X,targets,problems,freqs,File,Electrode):
    Cwt=np.abs(np.array(Cwtmatr))
    X=np.array(X)
    auto='auto'
    jet='jet'
    for i in range(len(problems)):
    # for i in range(2):
        cases=(np.unique(targets[problems[i]])).astype(int)
        figTitles=[['izq','der'],
                   ['0.1','0.2','0.4','0.8','1.6','3.2'],
                   ['3.2','1.6','0.8','0.4','0.2','0.1',
                    '-3.2','-1.6','-0.8','-0.4','-0.2','-0.1']]
                   
        if i==2:
            sbptls=[6,2]
        else:
            sbptls=[len(cases)+1,1]
        fig, axs=plt.subplots(sbptls[0],sbptls[1],figsize=(20,8))
        plt.suptitle([File[:-4],'Electrode: '+str(Electrode), problems[i]])
        if i!=2:
            for j in cases:
                exec('sum_'+str(j)+'=sum(targets[problems['+str(i)+']]=='+str(j)+')')
                exec('Cwt'+str(j)+'=sum(Cwt[targets[problems['+str(i)+']]=='+str(j)+'][:][:])/sum_'+str(j))
                exec('lfp'+str(j)+'=sum(X[targets[problems['+str(i)+']]=='+str(j)+'])/sum_'+str(j))
                exec('subfig=axs['+str(j)+'].imshow(Cwt'+str(j)+'[:,200:3000],aspect=''auto'',cmap=''jet'',clim=[np.min(Cwt'+str(j)+'[:,64:3500]),np.max(Cwt'+str(j)+'[:,64:3500])])')
                axs[j].set_ylabel('Frequency')
                axs[j].set_xlabel('Time')
                axs[j].set_title(figTitles[i][j])
                exec('axs['+str(j)+'].axvline(x=824,color=[1,0,0])')
                OriginalTicks=axs[j].get_yticks()
                axs[j].set_yticks(np.linspace(0,len(freqs),len(OriginalTicks)))
                axs[j].set_yticklabels(np.linspace(int(freqs[0]),int(freqs[-1]),len(OriginalTicks)).astype(int))
                #exec('axs['+str(j)+'].set_yticks(np.arange(1,10,1))')
            # exec('fig.colorbar(subfig, ax=axs[:-1])')
                exec('fig.colorbar(subfig, ax=axs['+str(j)+'])')
            for j in cases:
                exec('axs['+str(max(cases)+1)+'].plot(lfp'+str(j)+'[200:3000],color=[1,.6-.'+str(j+1)+',1-.'+str(j+1)+'],alpha=1)')
            exec('axs['+str(max(cases)+1)+'].axvline(x=824,color=[1,0,0])')
            exec('axs['+str(max(cases)+1)+'].set_xlim([0,2800])')
                
        else:
            cont=0
            for j in range(sbptls[0]):
                for k in range(sbptls[1]):
                    exec('sum_'+str(cont)+'=sum(targets[problems['+str(i)+']]=='+str(cont)+')')
                    exec('Cwt'+str(cont)+'=sum(Cwt[targets[problems['+str(i)+']]=='+str(cont)+'][:][:])/sum_'+str(cont))
                    exec('lfp'+str(cont)+'=sum(X[targets[problems['+str(i)+']]=='+str(cont)+'])/sum_'+str(cont))
                    exec('subfig=axs['+str(j)+']['+str(k)+'].imshow(Cwt'+str(j)+'[:,200:3000],aspect=''auto'',cmap=''jet'',clim=[np.min(Cwt'+str(cont)+'[:,64:3500]),np.max(Cwt'+str(cont)+'[:,64:3500])])')
                    ax=axs[j][k]
                    axs[j][k].set_ylabel('Frequency')
                    axs[j][k].set_xlabel('Time')
                    axs[j][k].set_title(figTitles[i][cont])
                    axs[j][k].axvline(x=824,color=[1,0,0])
                    OriginalTicks=axs[j][k].get_yticks()
                    axs[j][k].set_yticks(np.linspace(0,len(freqs),len(OriginalTicks)))
                    axs[j][k].set_yticklabels(np.linspace(int(freqs[0]),int(freqs[-1]),len(OriginalTicks)).astype(int))
                    exec('fig.colorbar(subfig, ax=ax)')
                    cont+=1
    
