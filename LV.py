import numpy as np
import argparse
import os
import time

import learner as ln
from LV_data import LVData

#default  
# LA:  
#args.filename='lv-l1'
#G:
#args.filename='lv-g1'
parser = argparse.ArgumentParser()
parser.add_argument('--device',type=int, default=0)
parser.add_argument('--filename', type=str, default='lv-l1')
parser.add_argument('--net_type', type=str, default='LA', choices=['LA', 'G', 'DFNN', 'ODE']) 

parser.add_argument('--train_num', type=int, default=75)
parser.add_argument('--test_num', type=int, default=75)
parser.add_argument('--data_step', type=float, default = 0.01)

parser.add_argument('--iterations', type=int, default=300000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_decay', default=1000)
parser.add_argument('--print_every', type=int, default=1000)

parser.add_argument('--LAlayers', type =int, default=9)
parser.add_argument('--LAlinlayers', type =int, default=3)
parser.add_argument('--LAsublinlayers', type =int, default=3)

parser.add_argument('--Glayers', type =int, default=3)
parser.add_argument('--Gsublayers', type =int, default=3)
parser.add_argument('--Gwidth', type=int, default=64)

args = parser.parse_args()

def main():
    device = 'cpu'
    h = args.data_step
    train_num = args.train_num
    test_num = args.test_num

    net_type = args.net_type 
    
    LAlayers = args.LAlayers
    LAlinlayers = args.LAlinlayers
    LAsublinlayers = args.LAsublinlayers
    order = [0,1,0,1,2,1,2,0,2]
    Glayers = args.Glayers
    Gsublayers = args.Gsublayers
    Gwidth = args.Gwidth
    
    activation = 'sigmoid'

    
    # training
    if not os.path.isdir('./outputs/'+args.filename): os.makedirs('./outputs/'+args.filename)
    f = open('./outputs/'+args.filename+'/output.txt',mode='w')
    f.write('net_type: '+net_type+ '  filename:' + args.filename
            +'\n\n'
            +'Start time  '+ time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())))
    f.close()

    criterion = 'MSE'
    data = LVData(h=h, train_num=train_num, test_num=test_num, add_h=False)

    if net_type == 'LA':
        net = ln.nn.LAVPNet(data.dim, order=order, h=0.001, layers=LAlayers, linlayers=LAlinlayers, 
                            sublinlayers=LAsublinlayers, activation=activation)
    elif net_type == 'G':
        net = ln.nn.GVPNet(data.dim, h=0.01, layers=Glayers, sublayers=Gsublayers, 
                           width=Gwidth, activation=activation)
    else: 
        raise NotImplementedError
            
    
    arguments = {
        'filename': args.filename,
        'data': data,
        'net': net,
        'criterion': criterion,
        'optimizer': 'adam',
        'lr': args.lr,
        'lr_decay': args.lr_decay,
        'iterations': args.iterations,
        'batch_size': None,
        'print_every': args.print_every,
        'save': True,
        'callback': None,
        'dtype': 'float',
        'device': device
    }

    
    ln.Brain.Init(**arguments)
    ln.Brain.Run()
    ln.Brain.Restore()
    ln.Brain.Output()
    
    if not os.path.isdir(args.filename): os.makedirs(args.filename)
    f = open(args.filename+'/args.txt',mode='w')
    f.write(str(args)+'\n')
    f.close()   
    
    steps = args.steps
    net=ln.Brain.Best_model()
    if isinstance(net, ln.nn.NeuralODE):
        flow_true = data.solver.flow(data.X_train_np[0][:-1], data.h, steps)
        flow_pred = net.predict(data.X_train[0][:-1], data.h, steps, keepinitx=True, returnnp=True)
    else:
        flow_true = data.solver.flow(data.X_train_np[0], data.h, steps)
        flow_pred = net.predict(data.X_train[0], steps, keepinitx=True, returnnp=True)
        
    np.save('./outputs/'+args.filename+'/flow_true.npy',flow_true)
    np.save('./outputs/'+args.filename+'/flow_pred.npy',flow_pred)
    
    if not os.path.isdir('./outputs/'+args.filename): os.makedirs('./outputs/'+args.filename)
    f = open('./outputs/'+args.filename+'/output.txt',mode='a')
    f.write('\n\n'+'Predict completion time  '+ time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())))
    f.write('\n\n'
            +'learning rate %s'%(args.lr)
            )
    f.close()
    
    f = open('./outputs/'+args.filename+'/args.txt',mode='w')
    f.write(str(args)+'\n')
    f.close()
    
if __name__ == '__main__': 
    main()
