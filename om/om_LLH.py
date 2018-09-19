#-------------------------------------------------------------
# This file gives examples of using the marginl probabilities calculated in
# om_functions to calculate likelihood
#-------------------------------------------------------------

import om_functions as om
import numpy as np
import scipy.stats as stats


"""
calculate the likelihood of the OM model. Assume a uniform foreground stellar density
and Gaussian foreground velocity distribution.
Parameter:
    model_param [list]: model parameters   -- ra [kpc]: anisotropy radius; 
                                           -- rho_s [M_sun/kpc^3]: normalization density;
                                           -- rs_s [kpc]: scale radius;
                                           -- al_s: transition slope index;
                                           -- be_s: outter slope index;
                                           -- ga_s: inner slope index;
    DM_param [list]: dark matter density parameters: 
                          -- rhos = 4*Pi*G *rho_d [(km/s)^2], 
                                     where rho_d is the normalization density of DM density
                          -- rs [kpc]:  the scale radius of the dark matter density
                          -- alpha: transition index of the alpha_beta_gamma density
                          -- beta : outer slope of the alpha_beta_gamma density
                          -- gamma: inner slope of the alpha_beta_gamma density
                          (# NOTE, required: gamma<3, beta>2, and alpha >0.)
    c1:  fraction of member stars
    mu2: mean of the foreground velocity distribution
    std2: standard deviation of the foreground velocity distribution

    Rp:   array; photometric data, projected radius; unit: kpc.
    R:    array; spectroscopic data, projected radius; unit: kpc.
    vz:   array; spectroscopic data, line-of-sight velocities; unit: km/s.
    vz0err: array; spectroscopic data, line-of-sight velocities observation error; unit: km/s.
Return: log-likelihood
"""
def LLH2c(model_param, DM_param, c1, mu2, std2, Rp, R, vz, vz0err):
    
    ra, rho_s, rs_s,  al_s, be_s, ga_s = model_param
    rhos, rs, alpha, beta, gamma = DM_param

    GQ, r200, rlim = om.GQ(model_param, DM_param, num_rsteps = 1e5, num_Qsteps = 1000)
    Rnorm = om.rhoLnorm(model_param) # stellar distribution normalization

    #to serves as foreground density function:
    maxR = max(Rp) #assume the foreground stars extend to the max projected radius of the dataset.
    comp2DF = 1./(np.pi*maxR*maxR) #constant foreground density, normalized


    if rlim>0: #rlim is set to be -99 for unphysical models.
      #LLH of spectroscopic data
      llv = 0
      for Ri,vzi, vzerri in zip(R,vz, vz0err):

        # conditional probabilities, P(vz|R), for member and foreground stars.
        p1 = om.VP_trapz(Ri,vzi, model_param, DM_param, GQ, r200, trapzN = 2000, rlim=rlim)
        stdev = (std2*std2 + vzerri*vzerri)**.5
        p2 = stats.norm(mu2,stdev).pdf(vzi) 

        #P(R)
        sigR1, ulimR = om.SigR_trapz(Ri, model_param, trapzN=2000, rlim=rlim)
        sigR1 = c1*(sigR1/Rnorm)
        sigR2 = (1-c1)*(comp2DF)

        pi = (p1*sigR1 + p2*sigR2) / (sigR1+sigR2) # total probability

        llv += np.log10(pi + 1e-100)

      #LLH of photometric data
      llR = 0
      for Rpi in Rp:
        if Rnorm > 0:
                sigR, ulimR = om.SigR_trapz(Rpi, model_param, trapzN=2000, rlim=rlim)
                sigR = c1*(sigR /Rnorm) + (1-c1)*(comp2DF)
                llR += np.log10(sigR + 1e-100)
        else:
                llR += np.log10(0 + 1e-100)

      loglike = llv + llR

    else: #i.e. rlim <= 0
      loglike = -100 * (len(Rp)+len(R))


    return loglike


#--------------- evalute the likelihood with Fornax data----------------

#import photometric and spectroscopic data to calculate the LLH
fspec = 'for_published_data_all'
fphoto = 'for_wpgs_des_isochrone.dat'
# load spec. data
x0, y0, vz0, vz0err = np.loadtxt(fspec, delimiter=' ', unpack=True, usecols=[0,1,2,3])
# load photo. data
xp,yp,Rp = np.loadtxt(fphoto, delimiter=' ', unpack=True, usecols=[2,3,4])
Rp = 147 * np.tan( Rp/ 60.*np.pi/180. ) #use a distance of d=147kpc to Fornax dSph

R = (x0*x0+y0*y0)**.5
vz = vz0

# test parameters
model_param = [1., 1.,  0.5, 2, 5, 0.5]
DM_param = [4000, 1., 1,3, .2]
c1 = .8
mu2 = 10
std2 = 100

llh = LLH2c(model_param, DM_param, c1, mu2, std2, Rp, R, vz, vz0err)

print 'test LLH: ', llh



