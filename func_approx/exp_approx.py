from math import sqrt, log, exp
from .affine_approx import AffineApprox

class ExpApprox:
    def __init__(self, x, y):
        coefs, self.corr = ExpApprox.coefs(x, y)
        self.a =coefs[0]
        self.b =coefs[1]
        self.x = x
        self.y = y
        self.errors = self.eval_errors()
        self.moy_err = AffineApprox.moy([abs(err) for err in self.errors])
        self.quad_err = sqrt(AffineApprox.moy([err**2 for err in self.errors]))
        
    def coefs(x, y):
        log_y = [log(yk) for yk in y]
        lin_coefs = AffineApprox.coefs(x, log_y)
        corr = AffineApprox.corr(x, log_y)
        
        return [exp(lin_coefs[0]), lin_coefs[1]], corr
    
    def predict(self, x):
        return self.a * exp(self.b * x)
    
    def predict_y(self, y):
        assert self.a !=0 and self.b != 0
        
        return (log(y/self.a)) / self.b
    
    def plot(self, ax, x_plot):
        y_plot = ([self.predict(x) for x in x_plot])
        ax.plot(x_plot, y_plot, label=self.__str__())
        ax.legend()
        
    def eval_errors(self):
        return [self.predict(xk) - yk for xk, yk in zip(self.x, self.y)]
        
    def __str__(self):
        return f"y = {self.a} * exp({self.b} * x)" 