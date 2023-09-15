import numpy as np

class FEFourier1DEq():
    def __init__(self, K, range, P):
        self._K = float(K)
        self._range = np.array(range)
        self._delta = (self.range[1]-self.range[0])/self.K

        self._P = int(P)

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
    
    @property
    def partial1(self):
        try:
            return self._partial1
        except AttributeError:
            tmp = np.zeros((self.dim, self.dim))
            tmp[:self.P, -self.P:] += np.diag(-(1+np.arange(self.P)))
            tmp[-self.P:, :self.P] += np.diag((1+np.arange(self.P)))
            self._partial1 = tmp*np.pi/self.delta
            return self._partial1

    @property
    def partial2(self):
        try:
            return self._partial2
        except AttributeError:
            tmp = np.zeros((self.dim, self.dim))
            tmp[:self.P, :self.P]   += np.diag(-(1+np.arange(self.P))**2)
            tmp[-self.P:, -self.P:] += np.diag(-(1+np.arange(self.P))**2)
            self._partial2 = tmp*(np.pi/self.delta)**2
            return self._partial2
    
    @property
    def tp_red(self):
        try:
            return self._tp_red
        except AttributeError:
            tmp = np.zeros((self.dim, self.dim, self.dim))
            sp = np.zeros((self.P+1, self.P, self.P))
            sn = np.zeros((self.P+1, self.P, self.P))
            cp = np.zeros((self.P+1, self.P, self.P))
            cn = np.zeros((self.P+1, self.P, self.P))

            for p1 in range(1, self.P+1):
                for p2 in range(1, self.P+1):
                    pp = p1 + p2
                    if pp <= self.P:
                        sp[pp-1, p1-1, p2-1]    += 1
                        cp[pp, p1-1, p2-1]      += 1

                    pn = p1 - p2
                    if pn > 0:
                        sn[pn-1, p1-1, p2-1]    += 1
                        cn[pn, p1-1, p2-1]      += 1
                    elif pn == 0:
                        cn[pn, p1-1, p2-1]      += 1
                    else:
                        apn = abs(pn)
                        sn[apn-1, p1-1, p2-1]   -= 1
                        cn[apn, p1-1, p2-1]     += 1


            tmp[-self.P-1:, :self.P, :self.P]   += (cn-cp)/2
            tmp[:self.P, :self.P, self.P]       += np.eye(self.P)
            tmp[:self.P+1, :self.P, -self.P:]   += (sp+sn)/2
            tmp[:self.P, self.P, :self.P]       += np.eye(self.P)
            tmp[self.P, self.P, self.P]         += 1.0
            tmp[-self.P:, self.P, -self.P:]     += np.eye(self.P)
            tmp[:self.P+1, -self.P:, :self.P]   += (sp-sn)/2
            tmp[-self.P:, -self.P:, self.P]     += np.eye(self.P)
            tmp[-self.P-1:, -self.P:, -self.P:] += (cn+cp)/2
            self._tp_red = tmp
            return self._tp_red
    
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