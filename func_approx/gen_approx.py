from abc import ABC, abstractmethod
from math import sqrt

class GenApprox(ABC):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        coefs, self.corr = self.coefs()
        self.a = coefs[1]
        self.b = coefs[0]
        self.abs_errors = self.eval_abs_errors()
        self.avg_abs_err = GenApprox.moy([abs(err) for err in self.abs_errors])
        self.avg_quad_err = sqrt(GenApprox.moy([err**2 for err in self.abs_errors]))
        self.rel_errors = self.eval_rel_errors()
        self.avg_rel_err = GenApprox.moy([abs(rel_err) for rel_err in self.rel_errors])
        self.avg_rel_err2 = sum([abs(err) for err in self.abs_errors]) / sum([abs(yk) for yk in self.y])
       
    def lin_coefs(x, y):
        var_x = GenApprox.var(x)
        assert  var_x != 0
        
        a = GenApprox.cov(x, y) / GenApprox.var(x)
        corr = GenApprox.corr(x, y)
        return [GenApprox.moy(y) - a * GenApprox.moy(x), a], corr
       
    @abstractmethod
    def coefs(self):
        pass
    
    @staticmethod
    def moy(datas):
        assert len(datas) >= 1
        
        return sum(datas) / len(datas)
    
    @staticmethod
    def var(datas):
        return GenApprox.moy([data**2 for data in datas]) - GenApprox.moy(datas)**2
    
    @staticmethod
    def cov(x, y):
        assert len(x) == len(y)
        
        return GenApprox.moy([xk * yk for xk, yk in zip(x, y)]) - GenApprox.moy(x) *  GenApprox.moy(y)
    
    @staticmethod
    def corr(x, y):
        denom = sqrt(GenApprox.var(x) * GenApprox.var(y))
        assert denom != 0
        
        return GenApprox.cov(x, y) / denom
    
    @abstractmethod
    def predict(self, x):
        pass
    
    @abstractmethod
    def predict_y(self, y):
        pass
    
    def plot(self, ax, x_plot):
        y_plot = ([self.predict(x) for x in x_plot])
        ax.plot(x_plot, y_plot, label=self.__str__())
        ax.legend()
        
    def eval_abs_errors(self):
        return [self.predict(xk) - yk for xk, yk in zip(self.x, self.y)]
    
    def eval_rel_errors(self):
        return [(self.predict(xk) - yk) / yk for xk, yk in zip(self.x, self.y)]
        
    @abstractmethod
    def __str__(self):
        pass
    