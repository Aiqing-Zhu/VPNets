import numpy as np
import torch

import learner as ln
from learner.integrator.rungekutta import ImMidpoint

true_A = torch.tensor([[-0.1, -2.0], [2.0, -0.1]])

def Lambda(y):
    
    return np.array([y[0]*(y[1]-y[2]), y[1]*(y[2]-y[0]), y[2]*(y[0]-y[1])])

class LVSolver:
    def __init__(self, order = 2, N=1):
        self.order = order
        self.N=N    
      
    def solve(self, x0, h):
        N=self.N
        if self.order == 2:
            for i in range(N):
                x0 = self.s2(x0, h / N)
            return x0
        elif self.order == 4:
            for i in range(N):
                x0 = self.s4(x0, h / N)
            return x0
        elif self.order == 6:
            for i in range(N):
                x0 = self.s6(x0, h / N)
            return x0
        else:
            raise NotImplementedError
    
        
    def lam1(self, y):
        return np.array([y[0]*(y[1]-y[2]),0, -y[2]*y[1]+ 0.5*y[2]**2])
    
    def lam2(self, y):
        return np.array([0,y[1]*(y[2]-y[0]), y[2]*y[0]- 0.5*y[2]**2])
    def Lambda(y):
    
        return np.array([y[0]*(y[1]-y[2]), y[1]*(y[2]-y[0]), y[2]*(y[0]-y[1])])
    def flow(self, x0, h, s):
        X = [x0]
        for i in range(s):
            X.append(self.solve(X[-1], h))
        return np.array(X)
        
    def s2(self, X, h):
        y1 = ImMidpoint(self.lam1, N=1, iteration=10).solver(X, h/2)
        y2 = ImMidpoint(self.lam2, N=1, iteration=10).solver(y1, h)
        y3 = ImMidpoint(self.lam1, N=1, iteration=10).solver(y2, h/2)
        return y3
    

    def s4(self, x0, h):
        r1 = 1 / (2 - 2 ** (1 / 3))
        r2 = - 2 ** (1/3) / (2 - 2 ** (1 / 3))
        x1 = self.s2(self.s2(self.s2(x0, r1 * h), r2 * h), r1 * h)
        return x1
    
    def s6(self, x0, h):
        r1 = 1 / (2 - 2 ** (1 / 5))
        r2 = - 2 ** (1 / 5) / (2 - 2 ** (1 / 5))
        x1 = self.s4(self.s4(self.s4(x0, r1 * h), r2 * h), r1 * h)
        return x1

def f2(y):
    return np.array([y[0]*(y[1]-2), y[1]*(1-y[0])])
class LVData(ln.Data):
    '''Data for learning 
    '''
    def __init__(self, x0=[5.,4.,6.], h=0.01, train_num=70, test_num=40, add_h=False):
        super(LVData, self).__init__()
        # self.f = f2
        self.solver = LVSolver(order=4, N=int(h/0.01)) #ImMidpoint(self.f, N=int(h/0.01), iteration=9)
        self.x0=x0
        self.h = h
        self.train_num = train_num
        self.test_num = test_num
        self.add_h = add_h
        self.__init_data()
        
    @property
    def dim(self):
        return 3
    
    def __generate_flow(self, h, num, operation='train'):
        # X = self.solver.flow(np.array(x0), h, num)
        if operation== 'train':
            
            X1,X2 = self.solver.flow(np.array([5.,4.1,5.9]), h, num),self.solver.flow(np.array([5.,3.9,6.1]), h, num)
            x, y = np.vstack((X1[:-1],X2[:-1])), np.vstack((X1[1:], X2[1:]))
        elif operation == 'test':
            X0, X1,X2 = self.solver.flow(np.array([5.,4.,6.]), h, num),self.solver.flow(np.array([5.2,4.,5.8]), h, num),self.solver.flow(np.array([4.9,4.,6.1]), h, num)
            x, y = np.vstack((X0[:-1], X1[:-1],X2[:-1])), np.vstack((X0[1:], X1[1:], X2[1:]))           
        else: raise ValueError
        if self.add_h:
            x = np.hstack([x, self.h * np.ones([x.shape[0], 1])])
        return x, y
    
    
    def __init_data(self):
        self.X_train, self.y_train = self.__generate_flow(self.h, self.train_num, operation='train')
        self.X_test, self.y_test = self.__generate_flow(self.h, self.test_num,operation='test')
    
    
    
    
    
    
    
    
    
    
    
    
