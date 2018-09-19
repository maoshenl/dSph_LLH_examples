from ctypes import cdll
import sys
import os.path
import ctypes
import time

lib = cdll.LoadLibrary("./sfw_function.so")


class Point(ctypes.Structure):
    _fields_ = [('x', ctypes.c_double), ('y', ctypes.c_double)]

    #def __repr__(self):
    #    return '({0}, {1})'.format(self.x, self.y)


class Params(ctypes.Structure):
    _fields_ = [('a', ctypes.c_double), ('d', ctypes.c_double),
		('e', ctypes.c_double), ('Ec', ctypes.c_double),
		('rlim', ctypes.c_double), ('b', ctypes.c_double),
		('q', ctypes.c_double), ('Jb', ctypes.c_double),
		('rhos', ctypes.c_double), ('rs', ctypes.c_double),
		('al', ctypes.c_double), ('be', ctypes.c_double), 
		('ga', ctypes.c_double)				]

def wrap_function(lib, funcname, restype, argtypes):
    """Simplify wrapping ctypes functions"""
    func = lib.__getattr__(funcname)
    func.restype = restype
    func.argtypes = argtypes
    return func

#sum = wrap_function(lib, 'sum', ctypes.c_double, [ctypes.c_double, ctypes.c_double])

p1=Params(2.0, -5.3, 2.5, 0.16, 1.5, +2.0, 6.9, 0.086, 4229.2, 0.694444, 1.00001, 3.00001, 1.0)
p0=Params(2.0, -5.3, 4.5, 0.16, 1.0, +2.0, 6.9, 0.086, 4.7*4229.2, 0.694444, 1.000001, 3.000001, 0.0)

#point = Point(1,3)
#print p1.rs

phi0 = wrap_function(lib, 'genphi0', ctypes.c_double,
                [ctypes.c_double, ctypes.c_double, ctypes.c_double,
		 ctypes.c_double, ctypes.c_double, ctypes.c_double])

fRV = wrap_function(lib, 'fRV', ctypes.c_double,
                [ctypes.c_double, ctypes.c_double, ctypes.c_double,
                 ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, Params])

fXvzw = wrap_function(lib, 'fXvzw', ctypes.c_double,
                [ctypes.c_double, ctypes.c_double, ctypes.c_double,
                 ctypes.c_double, ctypes.c_double, ctypes.c_double, Params])

fRvzw = wrap_function(lib, 'pRvz', ctypes.c_double,
                [ctypes.c_double, ctypes.c_double, Params, ctypes.c_double, ctypes.c_double])

#for b=2!
rhor = wrap_function(lib, 'rhor_3', ctypes.c_double,
                [ctypes.c_double, ctypes.c_double, ctypes.c_double, Params])

rhoR_4 = wrap_function(lib, 'pR', ctypes.c_double,
                [ctypes.c_double, Params, ctypes.c_double, ctypes.c_double])

norm = wrap_function(lib, 'norm', ctypes.c_double, [Params, ctypes.c_double, ctypes.c_double])

#for any b!
rhoR0 = wrap_function(lib, 'pR0', ctypes.c_double, 
		[ctypes.c_double, Params, ctypes.c_double, ctypes.c_double])
norm0 = wrap_function(lib, 'norm0', ctypes.c_double, [Params, ctypes.c_double, ctypes.c_double])



P0 = phi0(1e-10/p0.rs,  p0.rhos, p0.rs, p0.al, p0.be, p0.ga)
Plim = phi0(p0.rlim/p0.rs,  p0.rhos, p0.rs, p0.al, p0.be, p0.ga) - P0
Pr = phi0( 0.59160797831/p0.rs,  p0.rhos, p0.rs, p0.al, p0.be, p0.ga) - P0
xi,yi, zi, vxi,vyi, vzi = .1,.5, .3,  3,4,11
ri = (xi*xi+yi*yi+zi*zi)**.5
print 'P0, Pr, Plim: ', P0, Pr, Plim
#print 'fRVi: ', fRV(xi,yi, vxi,vyi,vzi, P0,Plim, p0)

fXvzwi = fXvzw(xi, yi, zi, vzi, P0, Plim, p0)
print 'c.fXvzwi: ', fXvzwi, (xi*xi+yi*yi+zi*zi)**.5, Pr

Ri = (xi*xi+yi*yi)**.5
fRvzwi = fRvzw(Ri, vzi, p0, P0, Plim)
print 'c.fRvzwi: ', fRvzwi


rhoR = rhoR_4(Ri, p0, P0, Plim)
norm = norm(p0, P0, Plim)
print 'c.rhoR_4: ', rhoR
print 'c.norm: ', norm

rhor = rhor(ri, P0, Plim, p0)
print 'c.rhor: ', rhor


t0 = time.time()
rhoR_t = rhoR0(Ri, p0, P0+0, Plim)
norm_t = norm0(p0, P0+0, Plim)
print 'c.rhoR_t: ', rhoR_t, rhoR, time.time()-t0
print 'c.norm0: ', norm_t, norm, time.time()-t0




