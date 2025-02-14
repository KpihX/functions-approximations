from math import log, exp
from .gen_approx import GenApprox

class ExpApprox(GenApprox):
    def __init__(self, x, y):
        super().__init__(x, y)
        
    def coefs(self):
        log_y = [log(yk) for yk in self.y]
        lin_coefs, corr = GenApprox.lin_coefs(self.x, log_y)
        
        return [exp(lin_coefs[0]), lin_coefs[1]], corr
    
    def predict(self, x):
        return self.b * exp(self.a * x)
    
    def predict_y(self, y):
        assert self.a != 0 and self.b != 0
        
        return (log(y/self.b)) / self.a
        
    def __str__(self):
        return f"y = {self.b} * exp({self.a} * x)" 