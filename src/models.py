import numpy as np

def FiveParam(Ecm, x_center, p1, p2, p3, p4, p5):
    """Nominal dN/dm density for the standard ATLAS dijet background."""
    x = x_center / Ecm
    nlog = np.log(x)
    return p1 * np.power((1.0 - x), p2) * np.power(x, (p3 + p4 * nlog + p5 * nlog * nlog))

def FiveParam_alt(Ecm, x_center, p1, p2, p3, p4, p5):
    """Alternative dN/dm density used to define the systematic uncertainty envelope."""
    x = x_center / Ecm
    nlog = np.log(x)
    return p1 * np.power((1.0 - x), p2) * np.power(x, (p3 + p4 * nlog + p5 / np.sqrt(x)))
