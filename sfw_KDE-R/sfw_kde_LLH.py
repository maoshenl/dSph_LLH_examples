import numpy as np
import scipy.stats as stats

import star_sampler as ssp
import sfw

#----------------------------------------------------------------------------------------------#
# This document gives an example of approximating the likelihood using Gaussian Kernel Density #
# Estimate of the sample points drawn from the model of interest, using StarSampler.           #
#----------------------------------------------------------------------------------------------#


"""
Approximate the likelihood of the SFW model. Assume a uniform foreground stellar density
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
def KDE_LLH(param, c1, mu2, std2, Rp, R, vz, vz0err):

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

    # Serve as photometric data foreground density:
    maxR = max(Rp) # Assume the foreground stars extend to the max projected radius of the dataset.
    comp2DF = 1./(np.pi*maxR*maxR) # constant foreground density, normalized


    #------------------------- sample and density estimate ---------------------------------------
    # Number of sample points to draw to approximate the density function. 
    # For example, here the samples are drawn using StarSampler.
    Nstars = 1e5 
    ssam = ssp.Sampler(myDF = sfw.sfw_fprob, sampler_input = sfw.sampler_input, model_param=param)
    x1,y1,z1,vx1,vy1,vz1 = ssam.sample(sample_method='impt', N = Nstars, steps=20, rfactor=10,
                filename=None, r_vr_vt=True)
    R1 = (x1*x1+y1*y1)**.5

    # Approximate the density using KDE
    kde  = stats.gaussian_kde((R1,vz1), bw_method='silverman')
    kdeR = stats.gaussian_kde(R1, bw_method='silverman')
    #----------------------------------------------------------------------------------------------


    # calculate conditional probability, p1 = P(vz|R).
    pRvz = kde((R,vz))/(2*np.pi*R) + 1e-100
    pR   = kdeR(R)/(2*np.pi*R) + 1e-100
    p1   = pRvz/pR

    # foreground velocity distribution
    stdev = (std2*std2 + vz0err*vz0err)**.5
    p2    = stats.norm(mu2,stdev).pdf(vz)

    
    # photometric data distribution.
    # member stars: sigR1; foreground stars: sigR2
    sigR1 = c1 * pR
    sigR2 = (1-c1) * comp2DF


    # probability of spectroscopic data
    p = ( p1*sigR1 + p2*sigR2 ) / (sigR1+sigR2)

    # probability of photometric data
    SigRp  = np.log10(  c1*kdeR(Rp)/(2*np.pi*Rp)  +  (1-c1)*(comp2DF) + 1e-100 )

    # log-likelihood
    loglike = np.sum( np.log10(p) )  +  np.sum( SigRp )

    return loglike









