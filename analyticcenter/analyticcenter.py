class AnalyticCenter(object):
    def __init__(self,  X, A_F, HX, algorithm=None, discrete_time=False):
        self.algorithm = algorithm
        self.X = X
        self.A_F = A_F
        self.HX = HX
        self.discrete_time = discrete_time