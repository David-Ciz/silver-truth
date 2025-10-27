from enum import Enum

class LossType(Enum):
    MSE = 1
    BCE = 2
    MSE_KL = 3
    BCE_KL = 4
    