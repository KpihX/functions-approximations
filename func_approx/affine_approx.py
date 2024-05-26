from .gen_approx import GenApprox

class AffineApprox(GenApprox):
    def __init__(self, x, y):
        super().__init__(x, y)
           
    def coefs(self):
        return GenApprox.lin_coefs(self.x, self.y)
    
    def predict(self, x):
        return self.a * x + self.b
    
    def predict_y(self, y):
        assert self.a != 0
        
        return (y - self.b) / self.a
    
    def __str__(self):
        return f"y = {self.a} * x + {self.b}" 