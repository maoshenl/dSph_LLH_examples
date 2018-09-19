#-------------------------------------------------------------------------
#  This file consists of functions to be used to evalue the probabilities
#  P(R) and P(R,v_z) for SFW b=2 model, and alpha_beta_gamma dark matter 
#  potential energy.
#-------------------------------------------------------------------------



import numpy as np
from scipy import integrate
import scipy.optimize
import scipy.special as ss


# General alpha_beta_gamma potential energy. Expression came from Mathematica by 
def genphi0(x, rhos, rs, alpha, beta, gamma):
        # integrating alpha_beta_gamma dark matter density. 
        # REQUIRED: (gamma<3, beta>2, and alpha >0)
        # Parameter:
        #      x = r/rs; r is the distance from center of the potential
        #               rs is the scale radius of the dark matter density
        #      rhos = 4*Pi*G *rho_d; unit: (km/s)^2 * (1/kpc^2), where rho_d is the 
        #                            normalization density of DM density
        #      gamma: inner slope of the alpha_beta_gamma density
        #      beta : outer slope of the alpha_beta_gamma density
        #      alpha: transition index of the alpha_beta_gamma density
        #      (   Required: (gamma<3, beta>2, and alpha >0)   )
        # Return: dark matter potential energy
        Ps = rhos * rs**2

        x0 = 10**-13
        alpha = alpha*1.0 #in case alpha is int 
        beta  = beta *1.0
        gamma = gamma*1.0

        #p1a = ss.hyp2f1((3.-gamma)/alpha, (beta*1.-gamma)/alpha, (3.+alpha-gamma)/alpha, -x0**alpha)
        p1b = ss.hyp2f1((3.-gamma)/alpha, (beta-gamma)/alpha, (3+alpha-gamma)/alpha,  -x**alpha)
        #I1  = ( x0**(3-gamma) * p1a - x**(3-gamma) * p1b ) / (x * (gamma - 3.))
        I1 = (0 - x**(3-gamma) * p1b ) / (x * (gamma - 3))

        p2  = ss.hyp2f1( (-2.+beta)/alpha,(beta-gamma*1.)/alpha,(-2.+alpha+beta)/alpha, -x**(-alpha))
        I2  = x**(2.-beta) * p2 / (beta -2.)

        ans1 = Ps * ( 1 - (I1 + I2) )

        return ans1


# Energy distribution of SFW model
def hE(x, E, param, P0, Plim):
        # Parameter:
        #      x = r/rs; r is the distance from center of the potential
        #               rs is the scale radius of the dark matter density
        #      E: energy per unit mass
        #      param [list]: model parameters
        #      P0: potential energy at r=0, to be used as zero-point
        #      Plim: potential energy at r=rlim, with P0 as the zero-point
        a,d,e, Ec, rlim, b, q, Jb, rhos, rs, alpha, beta, gamma = param

        Ps = rhos * rs**2     
        Ec = Ec * Ps

        N = 1.0 #set normalization constant to 1
        if E < Plim and E >= 0:
                h = N*(E**a) * ((E**q + Ec**q)**(d/q)) * ((Plim - E)**e)
        else:
                h = 0.0
        return h



# Assuming b=2, calculate join probability P(x,y,z,vz)
def fXvzw(x, y, z, vz, param, P0, Plim):
        a,d,e, Ec, rlim, b, q, Jb, rhos, rs, alpha, beta, gamma = param
        Ps = rhos * rs**2     

        xlim = rlim/rs
        r = (x*x+y*y+z*z)**.5
        Pr = genphi0(r/rs, rhos, rs, alpha,beta,gamma) - P0
        vesc = (2.*(Plim - Pr))**.5 if Plim>Pr else 0.

        if Pr>Plim:
                #print 'Plim < Pr: ', Plim, Pr, r, rlim
                return 0.0
        if vesc < abs(vz):
                return 0.0

        Jb   = Jb * rs * (Ps**0.5)

        c  = z/r    # cos(theta)
        c2 = c*c
        s2 = (1-c2) # sin(theta)^2
        s  = s2**.5
        def hgw_integrand(w):
                gw = np.pi*(2*s2*vz*vz + (1+c2)*w*w) #b=2
                E = w*w/2. + vz*vz/2. + Pr
                hEaux = hE(r/rs, E, param, P0, Plim)
                return hEaux * (2*np.pi + 1*((r/Jb)**b)*gw) * w

        wlim = (vesc*vesc - vz*vz)**.5
        result0 = integrate.quad(hgw_integrand, 0, wlim, epsrel = 1e-6, epsabs = 0)[0]

        return result0



# Assuming b=2, calculate join probability P(R,vz)
# marginalize fXvzw over z, to get join probability P(R,vz)
def pRvz(R,  vz, param, P0, Plim):
        # Parameter:
        #      R [kpc]: projected radius from center of the potential.
        #      vz: line-of-sight velocity
        #      param [list]: model parameters
        #      P0: potential energy at r=0, to be used as zero-point
        #      Plim: potential energy at r=rlim, with P0 as the zero-point
        # return: P(R,vz)
        a,d,e, Ec, rlim, b, q, Jb, rhos, rs, alpha, beta, gamma = param

        x = R/(2.**0.5)
        y = x
        def intg(z):
                return fXvzw(x, y, z, vz, param, P0, Plim)

        zlim = (rlim*rlim - R*R)**.5 if (rlim*rlim - R*R)>0 else 0.
        result = 2. * integrate.quad(intg, 0, zlim, epsrel = 1e-6, epsabs = 0)[0]
        return result



# stellar density profile, assuming b=2
def rhor_3(r, param, P0, Plim):
        # Parameter:
        #      r [kpc]: radius from center of the potential.
        #      param [list]: model parameters
        #      P0: potential energy at r=0, to be used as zero-point
        #      Plim: potential energy at r=rlim, with P0 as the zero-point
        # return: stellar density at r.
        a,d,e, Ec, rlim, b, q, Jb, rhos, rs, alpha, beta, gamma = param
        Ps = rhos * rs**2
        Jb = Jb * rs * (Ps**0.5)

        # NOTE: SFW model is formulate with Phi(r=0)=0, but that's not always true for 
	# alpha_beta_gamma potential, therefore we shift the potential down by subtracting Phi(0).
        Pr = genphi0(r/rs, rhos, rs, alpha,beta,gamma) - P0
        vmax = (2.*(Plim - Pr))**.5
        x = r/rs

        C2 = (r/Jb)**b * (np.pi**.5) * ss.gamma(1+0.5*b) / ss.gamma(0.5*(3+b))
        def Intg1(v):
                v2 = v*v
                E = 0.5*v2 + Pr
                return (2 + C2 * v**b) * v2 * hE(x, E, param, P0, Plim)

        I1 = integrate.quad(Intg1, 0, vmax, epsrel = 1e-6, epsabs = 0)[0]

        return I1 * 2*np.pi


# stellar surface density profile, assuming b=2. Not normalized
# marginalize rhor_3 over z, to get P(R)
def pR(R, param, P0, Plim):
        # Parameter:
        #      R [kpc]: Projected radius from center of the potential.
        #      param [list]: model parameters
        #      P0: potential energy at r=0, to be used as zero-point
        #      Plim: potential energy at r=rlim, with P0 as the zero-point
        # return: surface stellar density at R.
        a,d,e, Ec, rlim, b, q, Jb, rhos, rs, alpha, beta, gamma = param

        def Intg(z):
            if (z*z+R*R) > rlim*rlim:
                return 0
            else:
                r = (z*z+R*R)**.5
                result = rhor_3(r,  param, P0, Plim) 
                return result

        return 2*integrate.quad(Intg, 0, rlim, epsrel = 1e-6, epsabs = 0)[0]


# normalization constant of the model, assuming b=2
def norm(param, P0, Plim):
        # Parameter:
        #      param [list]: model parameters
        #      P0: potential energy at r=0, to be used as zero-point
        #      Plim: potential energy at r=rlim, with P0 as the zero-point
        # return: normalization constant
        a,d,e, Ec, rlim, b, q, Jb, rhos, rs, alpha, beta, gamma = param

        def Intg(r):
                result = rhor_3(r,  param, P0, Plim) *r*r
                return result

        ans = (4.*np.pi) * integrate.quad(Intg, 0, rlim, epsrel = 1e-6, epsabs = 0)[0]

        return ans



# Express SFW model f(E,J) = f(x,y,z,vx,vy,vz) through cross product (J = r x v)
def fXV(x1,y1,z1,vx,vy,vz, param, P0,Plim):
        # Parameter:
        #      x1 [kpc], x cartesian coordinates;
        #      y1 [kpc], y cartesian coordinates;
        #      z1 [kcp], z cartesian coordinates;
        #      vx [km/s], velocity in y-direction
        #      vy [km/s], velocity in x-direction 
        #      vz [km/s], line-of-sight velocity (z-direction)
        #      param, model parameters
        #      P0   [km/s]^2, potential energy at r=0, to be used as zero-point
        #      Plim [km/s]^2, potential energy at r=rlim, with P0 as the zero-point 
        # return: f(x,y,z,vx,vy,vz) # not normalized
        a,d,e, Ec, rlim, b, q, Jb, rhos, rs, alpha, beta, gamma = param

	# cross product (J = r x v)
        vi = y1*vz-z1*vy
        vj = -1*(x1*vz-z1*vx)
        vk = x1*vy-y1*vx
        J = (vi*vi + vj*vj + vk*vk)**.5

        r  = (x1*x1 + y1*y1 + z1*z1)**.5
        x  = r/rs

        Ps = rhos * rs**2   #unit (km/s)**2  
        Pr = genphi0(x, rhos, rs, alpha, beta, gamma) - P0
        #J  = abs(x*rs * vt)           #J = v * r*sin(theta)
        E  = (vx*vx + vy*vy + vz*vz)/2.0 + Pr # v*v/2 + Pr

        Ec   = Ec * Ps
        xlim = rlim / rs 
        Jb   = Jb * rs * (Ps**0.5) #*0.086

        if b <= 0:
                gJ1 = 1.0/(1 + (J/Jb)**-b)
        else:
                gJ1 = 1 + (J/Jb)**b

        N  = 1.0
        '''
        if E < Plim and E >= 0:
                hE1 = N*(E**a) * ((E**q + Ec**q)**(d/q)) * ((Plim - E)**e)
                #print "E: ", E
        else:
                hE1 = 0.0
        '''
        E = E * (E < Plim)*(E >= 0)
        hE1 = np.nan_to_num( N*(E**a)*((E**q + Ec**q)**(d/q))*((Plim - E)**e) )

        return hE1 * gJ1


# marginalize f(x,y,z,vx,vy,vz) over z, to obtain f(x,y,vx,vy,vz), 
# to calculate the likelihood of 5D data
def fRV(x,y, vx,vy,vz, param, P0,Plim):
        a,d,e, Ec, rlim, b, q, Jb, rhos, rs, alpha, beta, gamma = param

        RR = x*x+y*y

        def intg(z):
            return fXV(x,y,z,vx,vy,vz, param, P0,Plim)

        zlim = (rlim*rlim - RR)**.5 if rlim*rlim>RR else 0
        result = integrate.quad(intg, -zlim, zlim, epsrel=1.00e-05, epsabs=0)[0]

        return result




# calculate the dark matter mass enclosed within half-light radius, Rh, per rho_d, 
# where rho_d is the normalization constant of DM density.
def enclosed_mass_per_rho0(Rh, alpha, beta, gamma, rs):
        # Parameter:
        #      Rh [kpc]: half-light radius 
        #      gamma:    inner slope of the alpha_beta_gamma density
        #      beta :    outer slope of the alpha_beta_gamma density
        #      alpha:    transition index of the alpha_beta_gamma density
        #      rs:       scale radius of the dark matter density
        # return: mass enclosed within Rh per rho_d
        #  
        alpha = alpha*1.0 #in case alpha is int 
        beta  = beta *1.0
        gamma = gamma*1.0

        x0 = 10**-10
        rho0 = 1
        auxx = Rh/rs

        p1a = ss.hyp2f1((3-gamma)/alpha, (beta-gamma)/alpha, (3+alpha-gamma)/alpha, -x0**alpha)
        p1_b2 = ss.hyp2f1((3-gamma)/alpha,(beta-gamma)/alpha,(3+alpha-gamma)/alpha, -auxx**alpha)
        auxmass  = ( x0**(3-gamma) * p1a - auxx**(3-gamma) * p1_b2 ) / ((gamma - 3))
        mass_per_rho = 4*np.pi*auxmass * (rs**3) * rho0 #rho0 is just set to 1

        return mass_per_rho 












