from enum import Enum

class LossType(Enum):
    MSE = 1
    BCE = 2
    KLDIV = 3