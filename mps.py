import numpy as np
import scipy as sp

class MPS_value_bases():
    def __init__(self, n, ds, Ds):
        self.n = n

        if isinstance(ds, int):
            self.ds = np.array([ds] * self.n)
        elif isinstance(ds, (list, tuple)):
            self.ds = np.array(ds)

        if isinstance(Ds, int):
            self.Ds = np.array([Ds] * (self.n-1))
        elif isinstance(ds, (list, tuple)):
            self.Ds = np.array(Ds)

        self.random_init()

    def random_init(self):
        self.tensors = [np.random.uniform(-1, 1, [self.ds[0], self.Ds[0]])] +\
                       [np.random.uniform(-1, 1, list(shape)) for shape in zip(self.Ds[:-1], self.ds[1:-1], self.Ds[1:])] +\
                       [np.random.uniform(-1, 1, [self.Ds[-1], self.ds[-1]])]
        
    