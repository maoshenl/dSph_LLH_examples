#-------------------------------------------------------------
# This file gives examples of using the marginl probabilities calculated in
# sfw_functions to calculate likelihood
#-------------------------------------------------------------

import sfw_functions as fun
import scipy.stats as stats
import numpy as np

"""
calculate the likelihood of the SFW b=2 model. Assume a uniform foreground stellar density
and Gaussian foreground velocity distribution.
Parameter:
    param: list; param = [a,d,e, Ec, rlim, b, q, Jb, rhos, rs, alpha, beta, gamma] 
    c1:  fraction of member stars
    mu2: mean of the foreground velocity distribution
    std2: standard deviation of the foreground velocity distribution

    Rp:   array; photometric data, projected radius; unit: kpc.
    R:    array; spectroscopic data, projected radius; unit: kpc.
    vz:   array; spectroscopic data, line-of-sight velocities; unit: km/s.
    vz0err: array; spectroscopic data, line-of-sight velocities observation error; unit: km/s.
Return: log-likelihood
"""   

def LLH2c(param, c1, mu2, std2, Rp, R, vz, vz0err):

    
    a,d,e, Ec, rlim, b, q, Jb, rhos, rs, alpha, beta, gamma = param
    '''
    # NOTE: rhos = 4*Pi*G *rho_d, unit: (km/s)^2, where rho_d (unit: M_sun/kpc^3) is 
    #                       the normalization density of dark matter density
    # gamma: inner slope of the alpha_beta_gamma density
    # beta : outer slope of the alpha_beta_gamma density
    # alpha: transition index of the alpha_beta_gamma density
    # NOTE, required: gamma<3, beta>2, and alpha >0.
    # The rest of other parameters follow the convention in  Strigari et al. (2014) 
    #       (https://arxiv.org/abs/1406.6079), Table 1. '''

    
    # NOTE: SFW model is formulate with Phi(r=0)=0, but that's not always true for alpha_beta_gamma
    #       potential, therefore we shift the potential down by subtracting P(0).
    P0   = fun.genphi0(1e-10/rs, rhos, rs, alpha, beta, gamma) #calculate the potential zero-point
    Plim = fun.genphi0(rlim/rs, rhos, rs, alpha, beta, gamma) - P0 # potential at r=rlim

    Norm = fun.norm(param, P0, Plim)  # Density function normalization constant


    #serve as photometric data foreground density:
    maxR = max(Rp) #assume the foreground stars extend to the max projected radius of the dataset.
    comp2DF = 1./(np.pi*maxR*maxR) #constant foreground density, normalized


    #---------------photometric part of the likelihood ---------------------------
    llR = 0
    for Ri in Rp:
        if Norm>0:
            sigR1 = fun.pR(Ri, param, P0, Plim)
            sigR = c1*(sigR1/Norm) + (1-c1)*(comp2DF)
            llR += np.log10(sigR + 1e-100)
        else:
            llR += -100


    #-------------- spectroscopic part of the likelihood -------------------------
    llvz = 0
    for Ri,vzi, vzerri in zip(R,vz, vz0err):

	#foreground star velocity probability: p2
        stdev = (std2*std2 + vzerri*vzerri)**.5
        p2 = stats.norm(mu2,stdev).pdf(vzi) #normal mean/err

        #member star velocity probability: p1
        pR = fun.pR(Ri, param, P0, Plim) 	   #P(R)
        pRvz = fun.pRvz(Ri,  vzi, param, P0, Plim)  #P(R,vz)
        p1 = pRvz/pR if pR>0 else 1e-100             #conditional probability P(vz|R)

        sigR1 = pR
        sigR1 = c1*(sigR1/Norm) if Norm>0 else 1e-100
        sigR2 = (1-c1)*(comp2DF)
        pi = (p1*sigR1 + p2*sigR2) / (sigR1+sigR2) if (sigR1+sigR2)>0 else 1e-100

        llvz += np.log10(pi + 1e-100)

        loglike = llvz + llR

    return loglike    




#--------------- use ctypes-----------------------------------------------
import ctypes

#
lib = ctypes.cdll.LoadLibrary("./sfw_function.so")
class Params(ctypes.Structure):
    _fields_ = [('a', ctypes.c_double), ('d', ctypes.c_double),
                ('e', ctypes.c_double), ('Ec', ctypes.c_double),
                ('rlim', ctypes.c_double), ('b', ctypes.c_double),
                ('q', ctypes.c_double), ('Jb', ctypes.c_double),
                ('rhos', ctypes.c_double), ('rs', ctypes.c_double),
                ('al', ctypes.c_double), ('be', ctypes.c_double),
                ('ga', ctypes.c_double)                         ]

def wrap_function(lib, funcname, restype, argtypes):
    """Simplify wrapping ctypes functions"""
    func = lib.__getattr__(funcname)
    func.restype = restype
    func.argtypes = argtypes
    return func

pRvz = wrap_function(lib, 'pRvz', ctypes.c_double,
                [ctypes.c_double, ctypes.c_double, Params, ctypes.c_double, ctypes.c_double])

norm = wrap_function(lib, 'norm', ctypes.c_double, [Params, ctypes.c_double, ctypes.c_double])

pR = wrap_function(lib, 'pR', ctypes.c_double,
                [ctypes.c_double, Params, ctypes.c_double, ctypes.c_double])


def LLH(param, Rp, R, vz):

    # Same as LLH2c above except without fitting the foreground contaminants in 
    # bot photometric data and spectroscopic data. In addition, functions are
    # written using ctype to speedup the calculation.
    # IMPORTANT: the hypergeometric function (and the transformations applied to 
    #            extend the evaluable range) used is undefined at exactly alpha=1
    #            and beta=3; or at maybe some other integer values. 

    a,d,e, Ec, rlim, b, q, Jb, rhos, rs, alpha, beta, gamma = param
    alpha = alpha + .0000001 # the hypergeometric function used is undefined at exactly alpha=1
    beta  = beta  + .0000001 # the hypergeometric function used is undefined at exactly beta =3
    param0=Params(a,d,e, Ec, rlim, b, q, Jb, rhos, rs, alpha, beta, gamma)

    P0   = fun.genphi0(1e-10/rs, rhos, rs, alpha, beta, gamma)
    Plim = fun.genphi0(rlim/rs, rhos, rs, alpha, beta, gamma) - P0

    Norm = norm(param0, P0, Plim)
    llR = 0
    for Ri in Rp:
        if Norm>0:
            sigR = pR(Ri, param0, P0, Plim) # use the function written in c
            #sigR = fun.pR(Ri, param, P0, Plim) 
            llR += np.log10(sigR/Norm + 1e-100)
        else:
            llR += -100

    ll = 0
    for Ri,vzi in zip(R,vz):
        #f2 = pR(Ri, param0, P0, Plim) #NOTE!!
        f2 = fun.pR(Ri, param, P0, Plim)
        if f2 > 0:
            f0 = pRvz(Ri,  vzi, param0, P0, Plim) # use the function written in c
            #f0 = fun.pRvz(Ri,  vzi, param, P0, Plim)
            prob = f0/f2
            ll += np.log10(prob + 1e-100)
        else:
            ll += -100.

    loglike = ll + llR
    return loglike




#--------------- evalute the likelihood with Fornax data----------------

#import photometric and spectroscopic data
fspec = 'for_published_data_all'
fphoto = 'for_wpgs_des_isochrone.dat'

#load spec. data
x0, y0, vz0, vz0err = np.loadtxt(fspec, delimiter=' ', unpack=True, usecols=[0,1,2,3])
R = (x0*x0+y0*y0)**.5
vz = vz0

# load photo. data
xp,yp,Rp = np.loadtxt(fphoto, delimiter=' ', unpack=True, usecols=[2,3,4])
Rp = 147 * np.tan( Rp/ 60.*np.pi/180. )

# parameters
rs, alpha, beta, gamma = 1., 1.2, 3.2, .5
param = [2.0, -5.3, 4.5, 0.16, 1.0, +2.0, 6.9, 0.086, 1.*4229.2, rs, alpha, beta, gamma]
c1 = 1.0
mu2 = 10
std2 = 100

llh2c = LLH2c(param, c1, mu2, std2, Rp, R, vz, vz0err)
llh= LLH(param, Rp, R, vz)

print 'test LLH: ', llh2c, llh



