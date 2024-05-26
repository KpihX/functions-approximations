from math import log, exp
from .gen_approx import GenApprox

class PowApprox(GenApprox):
    def __init__(self, x, y):
        super().__init__(x, y)
        
    def coefs(self):
        log_y = [log(yk) for yk in self.y]
        log_x = [log(xk) for xk in self.x]
        lin_coefs, corr = GenApprox.lin_coefs(log_x, log_y)
         
        return [exp(lin_coefs[0]), lin_coefs[1]], corr
    
    def predict(self, x):
        return self.b * (x ** self.a) 
    
    def predict_y(self, y):
        assert self.a != 0 and self.b != 0
        
        return pow(y/self.b, 1/self.a)
        
    def __str__(self):
        return f"y = {self.b} * x ^ ({self.a})" 