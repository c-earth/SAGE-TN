import numpy as np

class FEFourier1DEq():
    def __init__(self, K, range, P):
        self._K = float(K)
        self._range = np.array(range)
        self._delta = (self.range[1]-self.range[0])/self.K

        self._P = P

    @property
    def K(self):
        return self._K
    
    @property
    def range(self):
        return self._range
    
    @property
    def min(self):
        return self.range[0]
    
    @property
    def max(self):
        return self.range[1]
    
    @property
    def delta(self):
        return self._delta
    
    @property
    def P(self):
        return self._P
    
    @property
    def dim(self):
        return 2*self.P+1
    
    def func(self, k, A):
        if len(A) != self.dim:
            raise ValueError(f'Expected {self.dim} Fourier components, but {len(A)} are given.')
        
        As = np.array(A[:self.P])
        An = np.array(A[self.P:-self.P])
        Ac = np.array(A[-self.P:])

        def genfunc(x):
            mask = (max(self.min, self.delta*(k-1)) < x) * (x < min(self.max, self.delta*(k+1)))

            return (An + \
                    np.sum(As*np.sin(np.pi/self.delta*np.arange(1, self.P + 1)*x.reshape((-1, 1))), axis = 1) + \
                    np.sum(Ac*np.cos(np.pi/self.delta*np.arange(1, self.P + 1)*x.reshape((-1, 1))), axis = 1)) * mask
        return genfunc
    
    