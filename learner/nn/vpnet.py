import torch
import torch.nn as nn

from .module import Module, StructureNN

class GradientModule(Module):
    '''Gradient volume preserving module.
    '''
    def __init__(self, dim, h=0.01, mode=0, width=10, activation='sigmoid'):
        super(GradientModule, self).__init__()
        self.dim = dim
        self.h=h
        self.width = width
        self.activation = activation
        self.mode = mode
        
        self.params = self.__init_params()
        
    def forward(self, x):
        i = int(self.mode)
        p = x[...,i:i+1] 
        q = torch.cat([x[...,:i], x[...,i+1:]],-1)
        subVP = self.act(q @ self.params['K1'] + self.params['b']) @ self.params['K2']
        return torch.cat([x[...,:i], p + subVP, x[...,i+1:]], -1)
    
    def __init_params(self):
        d = int(self.dim)
        params = nn.ParameterDict()
        params['K1'] = nn.Parameter((torch.randn([d-1, self.width]) ).requires_grad_(True))
        params['b'] = nn.Parameter((torch.randn([self.width]) ).requires_grad_(True))
        params['K2'] = nn.Parameter((torch.randn([self.width, 1])* self.h).requires_grad_(True))
        return params

class LinearModule(Module):
    '''Linear volume preserving module.
    '''
    def __init__(self, dim, h=0.01, layers = 1, sublayers=1):
        super(LinearModule, self).__init__()
        self.dim = dim
        self.h=h
        self.layers = layers 
        self.sublayers = sublayers
        self.params = self.__init_params() 
        
    def forward(self, x):
        d = int(self.dim)
        for i in range(self.layers):            
            for j in range(self.sublayers):
                x = self.__linmap(x, self.params['S{}{}'.format(i + 1, j + 1)], (i+j%2)%d)
        return (x + self.params['b'])
    
    def __init_params(self):
        '''Si is distributed N(0, 0.01), and b is set to zero.
        '''
        d = int(self.dim)
        params = nn.ParameterDict()
        for i in range(self.layers):
            for j in range(self.sublayers):
                params['S{}{}'.format(i + 1, j + 1)] = nn.Parameter(
                    # (torch.ones([d-1,1]).requires_grad_(True))
                     (torch.randn([d-1,1]) * self.h).requires_grad_(True)
                    )
        params['b'] = nn.Parameter((torch.zeros([d])).requires_grad_(True))
        return params
    
    def __linmap(self, x, S, i):
        if i==0:
            y = x[...,i:i+1] + x[...,i+1:]@S[i:]
        elif i==self.dim-1:
            y = x[...,i:i+1] + x[...,:i]@S[:i]
        else:
            y = x[...,i:i+1] + x[...,:i]@S[:i] + x[...,i+1:]@S[i:]   
        return torch.cat([x[...,:i], y, x[...,i+1:]], -1)

class ActivationModule(Module):
    '''Volume preserving activation module.
    '''
    def __init__(self, dim, h=0.01, mode=0, activation='sigmoid'):
        super(ActivationModule, self).__init__()
        self.dim = dim
        self.h = h
        self.mode=mode
        self.activation = activation
        self.params = self.__init_params()

        
    def forward(self, x):
        i = int(self.mode)
        p = x[...,i:i+1]
        q = torch.cat([x[...,:i], x[...,i+1:]],-1)
        add = self.act(q) @ self.params['a']
        return torch.cat([x[...,:i], p + add, x[...,i+1:]], -1)
    
    def __init_params(self):
        d = int(self.dim)
        params = nn.ParameterDict()
        params['a'] = nn.Parameter((torch.randn([d-1,1] )* self.h).requires_grad_(True))
        return params
    
class VPNet(StructureNN):
    def __init__(self):
        super(VPNet,self).__init__()
        self.dim=None
        
    def predict(self, xh, steps=1, keepinitx=False, returnnp=False):
        dim = xh.size(-1)
        size = len(xh.size())
        if dim == self.dim:
            pred = [xh]
            for _ in range(steps):
                pred.append(self(pred[-1]))
        else:
            x0, h = xh[..., :-1], xh[..., -1:] 
            pred = [x0]
            for _ in range(steps):
                pred.append(self(torch.cat([pred[-1], h], dim=-1)))
        if keepinitx:
            steps = steps + 1
        else:
            pred = pred[1:]
        res = torch.cat(pred, dim=-1).view([-1, steps, self.dim][2 - size:])
        return res.cpu().detach().numpy() if returnnp else res

class LAVPNet(VPNet):
    '''Volume preserving network composed by LinearModule and ActivationModule.
    
    '''
    def __init__(self, dim, order=[1,2,3], h=0.1, layers=2, linlayers=5, sublinlayers=3, activation='sigmoid'):
        super(VPNet, self).__init__()
        self.dim = dim
        self.order=order
        self.h=h
        self.layers = layers
        self.linlayers = linlayers
        self.sublinlayers = sublinlayers
        self.activation = activation    
        self.modus = self.__init_modules()
        
    def forward(self, x):
        for i in range(self.layers):
            LinM = self.modus['LinM{}'.format(i + 1)]
            NonM = self.modus['NonM{}'.format(i + 1)]
            x = NonM(LinM(x))
        return self.modus['LinMout'](x)
    
    def __init_modules(self):
        modules = nn.ModuleDict()
        for i in range(self.layers):
            mode = self.order[int(i%len(self.order))]%self.dim
            modules['LinM{}'.format(i + 1)] = LinearModule(self.dim, self.h, self.linlayers, self.sublinlayers)
            modules['NonM{}'.format(i + 1)] = ActivationModule(self.dim, self.h, mode, self.activation)
        modules['LinMout'] = LinearModule(self.dim, self.h, self.linlayers, self.sublinlayers)
        return modules

class GVPNet(VPNet):
    '''Volume preserving network composed by GradientModule.
    '''
    def __init__(self, dim, h=0.1, layers=5, sublayers=3, width=30, activation='sigmoid'):
        super(GVPNet, self).__init__()
        self.dim = dim
        self.h = h
        self.layers = layers
        self.sublayers = sublayers
        self.width = width
        self.activation = activation
        
        self.modus = self.__init_modules()
        
    def forward(self, x):
        for i in range(self.layers):
            for j in range(self.sublayers):
                subVP = self.modus['sub{}{}'.format(i+1,j+1)]
                x = subVP(x)
        return x
    
    def __init_modules(self):
        d = int(self.dim)
        modules = nn.ModuleDict()
        for i in range(self.layers):
            for j in range(self.sublayers):
                modules['sub{}{}'.format(i+1,j+1)] = GradientModule(
                    dim=self.dim, h=self.h, mode=(i+j%2)%d, width=self.width, activation=self.activation
                    )
        return modules
  
