import torch
import numpy as np
import matplotlib.pyplot as plt
from LV_data import LVData

def mse(a,b):
    return ((a-b)**2).sum(1)
xsize=18
legendsize=18
ticksize=18
titlesize=18
linewidth=2
hlinewidth = 1.5
def main():
    fig, ax= plt.subplots(2,3,figsize=(20,10))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None,
            wspace=0.2, hspace=0.2)
    
    predict(ax[0], net_type='g', index = '1', x0 = np.array([5,4,6]))
    predict(ax[1], net_type='l', index = '1', x0 = np.array([5,4,6]))

    ax[0,0].legend(loc='upper right', fontsize=legendsize)
    ax[0,1].legend(loc='upper right', fontsize=legendsize)
    ax[0,2].legend(loc='upper right', fontsize=legendsize)
    ax[1,0].legend(loc='upper right', fontsize=legendsize)
    ax[1,1].legend(loc='upper right', fontsize=legendsize)
    ax[1,2].legend(loc='upper right', fontsize=legendsize)
    xlim, ylim = 3.6,6.6
    ax[0,0].set_xlim(xlim, ylim)
    ax[0,0].set_ylim(xlim, ylim)
    ax[0,1].set_xlim(xlim, ylim)
    ax[0,1].set_ylim(xlim, ylim)
    ax[0,2].set_xlim(xlim, ylim)
    ax[0,2].set_ylim(xlim, ylim)
    ax[1,0].set_xlim(xlim, ylim)
    ax[1,0].set_ylim(xlim, ylim)
    ax[1,1].set_xlim(xlim, ylim)
    ax[1,1].set_ylim(xlim, ylim)
    ax[1,2].set_xlim(xlim, ylim)
    ax[1,2].set_ylim(xlim, ylim)
    
    predict(ax[0], net_type='g', index = '1', x0 = np.array([5.2,4,5.8]))
    predict(ax[0], net_type='g', index = '1', x0 = np.array([4.9,4,6.1]))
    
    predict(ax[1], net_type='l', index = '1', x0 = np.array([5.2,4,5.8]))    
    predict(ax[1], net_type='l', index = '1', x0 = np.array([4.9,4,6.1])) 
     
    fig.set_tight_layout(True)
    fig.savefig('LV_flow.pdf')

    
def predict(ax, net_type='l', index = '2', x0 = np.array([5,3.9,6.1])):
    e=0
    d=75
    sp, ep =0,d #ep<d, start prediction time and end prediction time

    
    data =LVData(x0, 0.01, 75,2)
    flow_true = data.solver.flow(x0, data.h, e+d)
    
    n = 'lv-'+net_type+index
    net = torch.load('outputs/'+n+'/model_best.pkl', map_location=torch.device('cpu'))
    flow_pred = net.predict(torch.FloatTensor(x0), steps=d, keepinitx=True, returnnp=True)
    

    if net_type =='l':
        label =  'LA-VPNet' 
        color = 'red'
    elif net_type =='g':
        label =  'R-VPNet' 
        color = 'blue'
    else: 
        raise ValueError
    
    
    ax[0].plot(flow_true[e+sp:e+ep,0],flow_true[e+sp:e+ep,1], color='grey',label ='Ground truth')
    ax[0].plot(flow_pred[sp:ep,0],flow_pred[sp:ep,1], color=color, linestyle= '--', label =label)
    
    ax[0].set_xlabel(r'$p$', fontsize=xsize)
    ax[0].set_ylabel(r'$q$', fontsize=xsize)
    
   
    ax[1].plot(flow_true[e+sp:e+ep,1],flow_true[e+sp:e+ep,2], color='grey',label ='Ground truth')
    ax[1].plot(flow_pred[sp:ep,1],flow_pred[sp:ep,2], color=color, linestyle= '--', label = label)
    
    ax[1].set_xlabel(r'$q$', fontsize=xsize)
    ax[1].set_ylabel(r'$r$', fontsize=xsize)
    
    ax[2].plot(flow_true[e+sp:e+ep,2],flow_true[e+sp:e+ep,0], color='grey',label ='Ground truth')
    ax[2].plot(flow_pred[sp:ep,2],flow_pred[sp:ep,0], color=color, linestyle= '--', label = label)
    
    ax[2].set_xlabel(r'$r$', fontsize=xsize)
    ax[2].set_ylabel(r'$p$', fontsize=xsize)
    return 0

if __name__=='__main__':
    main()
    
    
    
    
    
    
    
    
    
    