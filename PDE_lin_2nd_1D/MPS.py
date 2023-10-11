import numpy as np

class MPSFEFourier1DEq2func():
    pass

class MPSFEFourier1DEq():
    def __init__(self, FE, Delta, B, Us, Hcs, H0s, H1s, H2s):
        self._FE = FE
        self._Delta = Delta
        self._B = B
        self._Us = Us
        self._Hcs = Hcs
        self._H0s = H0s
        self._H1s = H1s
        self._H2s = H2s

    @property
    def FE(self):
        return self._FE
    
    @property
    def Delta(self):
        return self._Delta
        
    @property
    def B(self):
        return self._B
    
    @property
    def Us(self):
        return self._Us
    
    @property
    def Hcs(self):
        return self._Hcs
    
    @property
    def H0s(self):
        return self._H0s
    
    @property
    def H1s(self):
        return self._H1s
    
    @property
    def H2s(self):
        return self._H2s
    
    @property
    def U_prime(self):
        try:
            return self._U_prime
        except AttributeError:
            self._U_prime = [U.copy() for U in self.Us]
            return self._U_prime
        
    @property
    def H_tilde(self):
        try:
            return self._H_tilde
        except AttributeError:
            self._H_tilde = self.make_H_tilde()
            return self._H_tilde
        
    def multi_axes_stacking(Ts, axes):
        tmp_shape = Ts[0].shape
        num_axes = len(tmp_shape)
        for T in Ts:
            if num_axes != len(T.shape):
                raise ValueError('All tensors must have the same ranks.')
            tmp_shape[axes] += T.shape[axes]
        
        tmp = np.zeros(tmp_shape)
        start = np.zeros(num_axes)
        for T in Ts:
            stop = start.copy()
            stop[axes] += T.shape[axes]
            slices = [slice(first, second, 1) for first, second in zip(start, stop)]
            tmp[slices] = T
            start = stop
        return tmp
        
    def make_H_tilde(self):
        axes_ind = [[1]] + [[0, 2]] * (self.FE.K - 1) + [[0]]
        H_ind = [self.multi_axes_stacking([U, Hc], axes) for axes, U, Hc in zip(axes_ind, self.Us, self.Hcs)]

        If = np.expand_dims(np.eye(self.FE.dim), 2)
        Im = np.expand_dims(np.eye(self.FE.dim), (0, 3))
        Ib = np.expand_dims(np.eye(self.FE.dim), 0)
        Is = [If] + [Im] * (self.FE.K - 1) + [Ib]
        axes_dep = [[2]] + [[0, 3]] * (self.FE.K - 1) + [[0]]
        H_dep = [self.multi_axes_stacking([I, H0, H1, H2], axes) 
                 for axes, I, H0, H1, H2 
                 in zip(axes_dep, Is, self.H0s, self.H1s, self.H2s)]

    def make_U_tilde(self):
        pass

    def make_M_L(self):
        pass

    def make_M_R(self):
        pass

    def update_U_tilde(self):
        pass

    def update_M_L(self):
        pass

    def update_M_R(self):
        pass

    def update_U_prime(self):
        pass