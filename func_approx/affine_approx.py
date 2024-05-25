from math import sqrt

class AffineApprox:
    def __init__(self, x, y):
        coefs = AffineApprox.coefs(x, y)
        self.a = coefs[1]
        self.b = coefs[0]
        self.x = x
        self.y = y
        self.corr = AffineApprox.corr(x, y)
        self.errors = self.eval_errors()
        self.moy_err = AffineApprox.moy([abs(err) for err in self.errors])
        self.quad_err = sqrt(AffineApprox.moy([err**2 for err in self.errors]))
           
    def coefs(x, y):
        var_x = AffineApprox.var(x)
        assert  var_x != 0
        
        a = AffineApprox.cov(x, y) / AffineApprox.var(x)
        return [AffineApprox.moy(y) - a * AffineApprox.moy(x), a]
    
    def moy(datas):
        assert len(datas) >= 1
        
        return sum(datas) / len(datas)
    
    def var(datas):
        return AffineApprox.moy([data**2 for data in datas]) - AffineApprox.moy(datas)**2
    
    def cov(x, y):
        assert len(x) == len(y)
        
        return AffineApprox.moy([xk * yk for xk, yk in zip(x, y)]) - AffineApprox.moy(x) *  AffineApprox.moy(y)
    
    def corr(x, y):
        denom = sqrt(AffineApprox.var(x) * AffineApprox.var(y))
        assert denom != 0
        
        return AffineApprox.cov(x, y) / denom
    
    def predict(self, x):
        return self.a * x + self.b
    
    def predict_y(self, y):
        assert self.a != 0
        
        return (y - self.b) / self.a
    
    def plot(self, ax, x_plot):
        y_plot = ([self.predict(x) for x in x_plot])
        ax.plot(x_plot, y_plot, label=self.__str__())
        ax.legend()
        
    def eval_errors(self):
        return [self.predict(xk) - yk for xk, yk in zip(self.x, self.y)]
        
    def __str__(self):
        return f"y = {self.a} * x + {self.b}" 