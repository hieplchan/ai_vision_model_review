import time
import torch

class Timer:
    def __init__(self):
        self.clock = {}

    def start(self, key="default"):
        self.clock[key] = time.time()

    def end(self, key="default"):
        if key not in self.clock:
            raise Exception(f"{key} is not in the clock.")
        interval = time.time() - self.clock[key]
        del self.clock[key]
        print('[TIMER] {} take {:06.3f} ms'.format(key, interval*1000))
        return interval

timer = Timer()

def test(a,b,cmp,cname=None):
    if cname is None: cname=cmp.__name__
    assert cmp(a,b),f"{cname}:\n{a}\n{b}"

def near(a,b):
    return torch.allclose(a, b, rtol=1e-3, atol=1e-5)

def test_near(a,b):
    test(a,b,near)

def test_near_zero(a, tol=1e-3):
    assert a.abs()<tol, f"Near zero: {a}"
