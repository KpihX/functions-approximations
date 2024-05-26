from math import log, exp
from .gen_approx import GenApprox

class LogApprox(GenApprox):
    def __init__(self, x, y):
        super().__init__(x, y)
        
    def coefs(self):
        log_x = [log(xk) for xk in self.x]
        lin_coefs, corr = GenApprox.lin_coefs(log_x, self.y)
        
        return [lin_coefs[0], lin_coefs[1]], corr
    
    def predict(self, x):
        return self.a * log(x) + self.b
    
    def predict_y(self, y):
        assert self.a != 0
        
        return exp((y - self.b) / self.a)
        
    def __str__(self):
        return f"y = {self.a} * log(x) + {self.b}"
