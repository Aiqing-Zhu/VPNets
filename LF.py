import argparse
import os
import time
import torch

import learner as ln
from LF_data import LFData

#default  
# LA:  
#args.filename='lf-l'     
# args.lr = 0.01
# args.lr_decay=100
# args.iterations = 800000
#G:
#args.filename='lf-g'   
# args.lr = 0.001
# args.lr_decay=100
# args.iterations = 500000

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=int, default=0)
parser.add_argument('--filename', type=str, default='lf-l')
parser.add_argument('--net_type', type=str, default='LA', choices=['LA', 'G', 'DFNN', 'NODE']) 

parser.add_argument('--train_num', type=int, default=100)
parser.add_argument('--test_num', type=int, default=100)
parser.add_argument('--data_step', type=float, default = 0.5)

parser.add_argument('--iterations', type=int, default=800000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_decay', default=100)
parser.add_argument('--print_every', type=int, default=1000)

parser.add_argument('--LAlayers', type =int, default=12)
parser.add_argument('--LAlinlayers', type =int, default=4)
parser.add_argument('--LAsublinlayers', type =int, default=3)

parser.add_argument('--Glayers', type =int, default=4)
parser.add_argument('--Gsublayers', type =int, default=3)
parser.add_argument('--Gwidth', type=int, default=64)


args = parser.parse_args()

def main():
    if torch.cuda.is_available():
        device = 'gpu'
        torch.cuda.set_device(args.device)
    else: 
        device ='cpu'
    # data
    x0 = [0.1,1,1,0.2]
    h = args.data_step
    train_num = args.train_num
    test_num = args.test_num

    #network
    net_type = args.net_type 
    activation='sigmoid'
    LAlayers = args.LAlayers
    LAlinlayers = args.LAlinlayers
    LAsublinlayers = args.LAsublinlayers
    order = [0,1,0,1,2,1,2,3,2,3,0,3]
    Glayers = args.Glayers
    Gsublayers = args.Gsublayers
    Gwidth = args.Gwidth
    
    # training
    add_h = False
    criterion = 'MSE'
    data = LFData(x0, h, train_num, test_num, add_h)
    
    if net_type == 'LA':
        net = ln.nn.LAVPNet(data.dim, order=order, h=0.001, layers=LAlayers, linlayers=LAlinlayers, 
                            sublinlayers=LAsublinlayers, activation=activation)
    elif net_type == 'G':
        net = ln.nn.GVPNet(data.dim, h=0.01, layers=Glayers, sublayers=Gsublayers, 
                           width=Gwidth, activation=activation)          
    

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
    
    if not os.path.isdir('./outputs/'+args.filename): os.makedirs('./outputs/'+args.filename)
    f = open('./outputs/'+args.filename+'/output.txt',mode='w')
    f.write('net_type: '+net_type+ '  filename:' + args.filename
            +'\n\n'
            +'Start time  '+ time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())))
    f.write('\n\n'
        +'activation: '+activation+'\n'
        +'train_num %s'%(args.train_num)+'\n'
        +'test_num %s'%(args.test_num)+'\n'
        +'data_step %s'%(args.data_step)+'\n'
        +'iterations %s'%(args.iterations)+'\n'
        +'learning rate %s'%(args.lr)+'\n'
        +'lr_decay%s'%(args.lr_decay)+'\n'
        )
    f.close()
    
    ln.Brain.Init(**arguments)
    ln.Brain.Run()
    ln.Brain.Restore()
    ln.Brain.Output()
    
    if not os.path.isdir(args.filename): os.makedirs(args.filename)
    f = open(args.filename+'/args.txt',mode='w')
    f.write(str(args)+'\n')
    f.close()    
    if not os.path.isdir('./outputs/'+args.filename): os.makedirs('./outputs/'+args.filename)
    f = open('./outputs/'+args.filename+'/output.txt',mode='a')
    f.write('\n\n'+'Predict completion time  '+ time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())))

    f.close()

    f = open('./outputs/'+args.filename+'/args.txt',mode='w')
    f.write(str(args)+'\n')
    f.close()
    
if __name__ == '__main__':
    main()
