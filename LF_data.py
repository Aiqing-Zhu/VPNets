import numpy as np
import learner as ln

class LFSolver:
    '''Solve Lorentz force equation in electromagnetic field
        dx/dt = v
        dv/dt = q/m (E + v cross B)
        by Boris method
        xn+1 = xn + vn+1
        vn+1 = vn + q/m(En+ [vn+0.5] cross Bn)
    '''
    def __init__(self, N = 10):
        self.N = N
    
    def solve(self, x0, h):
        N = self.N
        for i in range(N):
            x0 = LFSolver.Boris2(x0, h / N)
        return x0
        
    def flow(self, x0, h, s):
        X = [x0]
        for i in range(s):
            X.append(self.solve(X[-1], h))
        return np.array(X)

    @staticmethod 
    def Boris(y0, h):
        '''Boris for 3-dimension system, dim =6
        '''
        def MagneticField(x):
            return np.array([0,0,np.sqrt(x[0]**2+x[1]**2)])
        def ElectricField(x):
            R = np.sqrt(x[0]**2+x[1]**2)
            return np.array([x[0],x[1],0])/(100*R**3)           
        def Omega(x,h):      
            B=MagneticField(x)*h/2
            return np.array([[0,-B[2], B[1]],
                             [ B[2],0, B[0]],
                             [-B[1],-B[0],0]])    
        def inverIOmega(x,h):
            B=MagneticField(x)*h/2
            return np.array([[1+B[0]**2, B[2]-B[0]*B[1], -B[1]-B[2]*B[0]],
                             [-B[2]-B[0]*B[1], 1+B[1]**2, -B[0]+B[1]*B[2]],
                             [B[1]-B[0]*B[2],B[0]+B[1]*B[2], 1+B[2]**2]])/(1+B[0]**2+B[1]**2+B[2]**2)
        def R(x,h):
            I = np.identity(3)      
            return np.matmul(inverIOmega(x,h),I-Omega(x,h))      
        x0 = y0[:3]
        v0 = y0[3:]
        v =  v0 @  R(x0,h).T+ h*ElectricField(x0) @ inverIOmega(x0,h).T 
        x = x0 + h*v
        return np.hstack((x,v))
    
    @staticmethod 
    def Boris2(y0, h):
        '''Boris for 2-dimension system, dim=4
        '''
        def MagneticField(x):
            return np.array([0,0,np.sqrt(x[0]**2+x[1]**2)])
        def ElectricField(x):
            R = np.sqrt(x[0]**2+x[1]**2)
            return np.array([x[0],x[1]])/(100*R**3)           
        def Omega(x,h):      
            B=MagneticField(x)*h/2
            return np.array([[0,-B[2]],
                             [B[2],0],])    
        def inverIOmega(x,h):
            B=MagneticField(x)*h/2
            return np.array([[1, B[2]],
                             [-B[2], 1]])/(1+B[2]**2)
        def R(x,h):
            I = np.identity(2)      
            return np.matmul(inverIOmega(x,h),I-Omega(x,h))      
        x0 = y0[:2]
        v0 = y0[2:]
        v =  v0 @  R(x0,h).T+ h*ElectricField(x0) @ inverIOmega(x0,h).T 
        x = x0 + h*v
        return np.hstack((x,v))
   
        
class LFData(ln.Data):
    '''Data for learning the Lorentz force equation in electromagnetic field
    '''    
    def __init__(self, x0, h, train_num, test_num, add_h=False):
        super(LFData, self).__init__()
        self.solver = LFSolver(N=int(h/0.01))
        self.x0 = x0
        self.h = h
        self.train_num = train_num
        self.test_num = test_num
        self.add_h = add_h
        self.__init_data()
    
    @property
    def dim(self):
        return 4
    
    def __generate_flow(self, x0, h, num):      
        X = self.solver.flow(x0, h=h, s=num)
        x, y = X[:-1], X[1:]
        
        if self.add_h:
            x = np.hstack([x, self.h * np.ones([x.shape[0], 1])])
        return x, y
    
    def __init_data(self):
        self.X_train, self.y_train = self.__generate_flow(self.x0, self.h, self.train_num)
        self.X_test, self.y_test = self.__generate_flow(self.y_train[-1], self.h, self.test_num)

def main():
   import matplotlib.pyplot as plt
   x0 = [0.1,1,1,0.2]
   b,e=0,80*5
   data = LFData(x0, 0.1, 2, 2, False)
   flow_true = data.solver.flow(data.X_train_np[0], data.h, 800) 
    
   plt.plot(flow_true[b:e:5,2],flow_true[b:e:5,3],'r')
   
   data = LFData(x0, 0.1, 2, 2, False)
   data.solver=LFSolver(N=20)
   plt.show()
   return 0

if __name__ == '__main__':
   
   main()