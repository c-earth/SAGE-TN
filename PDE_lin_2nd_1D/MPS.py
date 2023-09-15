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
            self._U_prime = []
            return self._U_prime
    
    @property
    def Y(self):
        try:
            return self._Y
        except AttributeError:
            self._Y = [U - self.Delta*Hc for U, Hc in zip(self.Us, self.Hcs)]
            return self._Y
    
    @property
    def h(self):
        try:
            return self._h
        except AttributeError:
            self._h = [self.Delta *\
                       (
                        np.einsum('ijk,k->ij', self.FE.tp_red, H0) +\
                        np.einsum('ilk,k,lj->ij', self.FE.tp_red, H1, self.FE.partial1) +\
                        np.einsum('ilk,k,lj->ij', self.FE.tp_red, H2, self.FE.partial2)
                       )
                       for H0, H1, H2 in zip(self.H0s, self.H1s, self.H2s)]
            return self._h
        
    def 