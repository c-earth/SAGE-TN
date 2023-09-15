import numpy as np

class MPSFEFourier1DEq():
    def __init__(self, FE, B, Ts, As = None):
        self._FE = FE
        self._B = B
        self._matrices = self.make_matrices(As)
    
    @property
    def FE(self):
        return self._FE
    
    @property
    def B(self):
        return self._B
    
    def make_matrices(self, As):
        if As == None:
            matrices = [np.random.rand()] + [] + []
        
        else:
            pass

        return matrices
        
    
    


class MPSFourier1DEq2func():
    pass