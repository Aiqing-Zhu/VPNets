import torch
import numpy as np
import matplotlib.pyplot as plt
from LF_data import LFData


def main():
    fig, ax= plt.subplots(3,2,figsize=(14,17))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=0.25, hspace=0.3)
    x0 = [0.1,1,1,0.2]
    e=100
    d=150    
    data = LFData(x0, 0.5, 2, 2, False)
    flow_true = data.solver.flow(data.X_train_np[0], data.h, e+d)
    df = int((d-1)*data.h/0.01)+1
    flow= data.solver.flow(flow_true[e], 0.01, df)
    

    x,y=0,1
    ax[0,0].plot(flow[:df,x],flow[:df,y], linestyle= '--',
                 color = 'grey', linewidth =1, label='Ground truth',zorder = 0)
    ax[0,0].scatter(flow_true[e:e+d,x],flow_true[e:e+d,y], 
                    color = 'grey', linewidth =1, label='Exact position',zorder = 0)   
    
    ax[0,1].plot(flow[:df,x],flow[:df,y], linestyle= '--',
                 color = 'grey', linewidth =1, label='Ground truth',zorder = 0)
    ax[0,1].scatter(flow_true[e:e+d,x],flow_true[e:e+d,y], 
                    color = 'grey', linewidth =1, label='Exact position',zorder = 0)
    

    x,y=2,3
    ax[1,0].plot(flow[:df,x],flow[:df,y], linestyle= '--',
                 color = 'grey', linewidth =1, label='Ground truth',zorder = 0)
    ax[1,0].scatter(flow_true[e:e+d,x],flow_true[e:e+d,y], 
                    color = 'grey', linewidth =1, label='Exact velocity',zorder = 0)   
    
    ax[1,1].plot(flow[:df,x],flow[:df,y], linestyle= '--',
                 color = 'grey', linewidth =1, label='Ground truth',zorder = 0)
    ax[1,1].scatter(flow_true[e:e+d,x],flow_true[e:e+d,y], 
                    color = 'grey', linewidth =1, label='Exact velocity',zorder = 0)
    
    ax[2,0].set_ylim(-0.0005, 0.0045)
    
    predict(ax[0,1], ax[1,1], ax[2,0], ax[2,1], data, flow_true, e,d, 
            net_type='l',color='red', marker='x', zorder =1)
    predict(ax[0,0], ax[1,0], ax[2,0], ax[2,1], data, flow_true, e,d, 
            net_type='g',color='blue', marker='+', zorder =1)

    xsize=15
    legendsize=12
    titlesize=15

    ax[0,0].set_title('Predicted position',fontsize=titlesize,loc='left')
    ax[0,0].set_xlabel(r'$x_1$', fontsize=xsize)
    ax[0,0].set_ylabel(r'$x_2$', fontsize=xsize)
    ax[0,0].legend(loc='center',fontsize=legendsize)
    
    ax[0,1].set_title('Predicted position',fontsize=titlesize,loc='left')
    ax[0,1].set_xlabel(r'$x_1$', fontsize=xsize)
    ax[0,1].set_ylabel(r'$x_2$', fontsize=xsize)
    ax[0,1].legend(loc='center',fontsize=legendsize)
    
    ax[1,0].set_title('Predicted velocity',fontsize=titlesize,loc='left')
    ax[1,0].set_xlabel(r'$v_1$', fontsize=xsize)
    ax[1,0].set_ylabel(r'$v_2$', fontsize=xsize)
    ax[1,0].legend(loc='center',fontsize=legendsize)
    
    ax[1,1].set_title('Predicted velocity',fontsize=titlesize,loc='left')
    ax[1,1].set_xlabel(r'$v_1$', fontsize=xsize)
    ax[1,1].set_ylabel(r'$v_2$', fontsize=xsize)
    ax[1,1].legend(loc='center',fontsize=legendsize)
    
    ax[2,0].set_title('Conservation of energy',fontsize=titlesize,loc='left')
    ax[2,0].set_xlabel('Time', fontsize=xsize)
    ax[2,0].set_ylabel('Energy error', fontsize=xsize)
    ax[2,0].legend(loc='upper left',fontsize=legendsize)
    
    ax[2,1].set_title('Global error',fontsize=titlesize,loc='left')
    ax[2,1].set_xlabel('Time', fontsize=xsize)
    ax[2,1].set_ylabel('Error', fontsize=xsize)
    ax[2,1].legend(loc='upper left',fontsize=legendsize)
    fig.savefig('LF.pdf',dpi=300, bbox_inches='tight')
    return 0    

def predict(ax1, ax2, ax_energy, ax_error, data, flow_true, e, d, net_type='l',color='red', marker='x', zorder=1):
    
    nettype = {'l': 'LA-VPNet',
               'g': 'R-VPNet',
          }
    netlocal = {'l': 'lf-'+net_type,
               'g': 'lf-'+net_type,
          }
    label = nettype[net_type]
    n = netlocal[net_type]
    net = torch.load('outputs/'+n+'/model_best.pkl', map_location=torch.device('cpu'))#outputs/'+n+'/model_best.pkl
    flow_pred = net.predict(torch.FloatTensor(flow_true[e]), d, keepinitx=True, returnnp=True)

    x,y =0,1
    ax1.scatter(flow_pred[:d,x],flow_pred[:d,y], 
                color=color, marker=marker, label = label, zorder=zorder)
    x,y =2,3
    ax2.scatter(flow_pred[:d,x],flow_pred[:d,y], 
                color=color, marker=marker, label = label, zorder=zorder)   
    
    t=(np.arange(e,e+d,1))*data.h 
    En = np.abs(energy(flow_pred[:d])-energy(flow_pred[0:1]))
    ax_energy.plot(t,En, color=color,marker=marker,label=label+'s',zorder=0)
    
    error=np.vstack((mse(flow_pred[:d],flow_true[e:e+d]),
                        mse(flow_pred[:d],flow_true[e:e+d]),
                        mse(flow_pred[:d],flow_true[e:e+d])))
    error_mean = error.mean(0)
    ax_error.plot(t, error_mean, color=color,marker=marker,label=label+'s')
    return 0

def mse(a,b):
    return ((a-b)**2).sum(1)
def energy(flow): 
    return (flow[:,2:]**2)@np.ones((2,1))/2 + 1/(100*np.sqrt((flow[:,0:2]**2)@np.ones((2,1))))
   

if __name__=='__main__':
    main()
    
    
    
    
    
    
    
    
    
    
