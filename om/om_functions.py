import numpy as np
import random as rand
import scipy.special as ss
import time
import scipy.optimize
from scipy import integrate
import warnings
from scipy.interpolate import PchipInterpolator

#-------------------------------------------------------------------------------------
# The following functions are used to calculate the likelihood of the Osipkov-Merritt
# model. Two sets of the common parameters used are model_param and dark matter DM_param
#    model_param [list]: model parameters   -- ra [kpc]: anisotropy radius; 
#                                           -- rho_s [M_sun/kpc^3]: normalization density;
#                                           -- rs_s [kpc]: scale radius;
#                                           -- al_s: transition slope index;
#                                           -- be_s: outter slope index;
#                                           -- ga_s: inner slope index;
#    DM_param [list]: dark matter density parameters: 
#                          -- rhos = 4*Pi*G *rho_d [(km/s)^2 * 1/kpc^2], 
#                                     where rho_d is the normalization density of DM density
#                          -- rs [kpc]:  the scale radius of the dark matter density
#                          -- alpha: transition index of the alpha_beta_gamma density
#                          -- beta : outer slope of the alpha_beta_gamma density
#                          -- gamma: inner slope of the alpha_beta_gamma density
#
# The calculation assumes that contribution from the stellar mass is negligible. Potential 
# energy is calculated with only dark matter density. If it is desired to include stellar mass
# one needs to re-calculate r200 and OMgenphi appropriate (using OMgenphi2 () and getR200b()).
#-------------------------------------------------------------------------------------


# General alpha_beta_gamma potential energy. Expression came from Mathematica by 
# integrating alpha_beta_gamma dark matter density. 
# REQUIRED: (gamma<3, beta>2, and alpha >0)
def OMgenphi(r, rtrunc, rhos, rs, alpha, beta, gamma):
        # Parameter:
        #      r  [kpc]: the distance from center of the potential
        #      rtrunc [kpc]: truncation radius, where the density is set to zero beyond that point,
        #                    so the potential is zero at infinity.
        #      For rhos, rs, alpha, beta, and gamma, see above.
        #      (   Required: (gamma<3, beta>2, and alpha >0)   )
        # Return: dark matter potential energy
        x0 = 10**-12
        xlim = (rtrunc + 0) / rs
        alpha = alpha*1.0 #in case alpha is int 
        beta  = beta *1.0
        gamma = gamma*1.0

        x = r/rs #+ 1e-10
        Ps = rhos * rs**2

        x2 = x
        try:
            x2[x2>xlim] = xlim
        except:
            x2 = x2 if x2<xlim else xlim

        p2a = ss.hyp2f1( (2.-gamma)/alpha,(beta-gamma*1.)/alpha, (2.+alpha-gamma)/alpha, -x2**(alpha) )
        p2b = ss.hyp2f1( (2.-gamma)/alpha,(beta-gamma*1.)/alpha, (2.+alpha-gamma)/alpha, -xlim**(alpha) )
        I2  = (x2**(2-gamma) * p2a - xlim**(2-gamma) * p2b) / (gamma - 2)

        p1a = ss.hyp2f1((3.-gamma)/alpha, (beta*1.-gamma)/alpha, (3.+alpha-gamma)/alpha, -x0**alpha)
        p1b = ss.hyp2f1((3.-gamma)/alpha, (beta-gamma)/alpha, (3+alpha-gamma)/alpha,  -x2**alpha)
        I1  = ( x0**(3-gamma) * p1a - x2**(3-gamma) * p1b ) / ((r/rs) * (gamma - 3.))

        PE = Ps * ( 0 - (I1 + I2) ) 

        return PE

# same as OMgenphi as define above, except the potential energy also has contribution 
# from stellar components
def OMgenphi2(r, rtrunc, model_param, DM_param):
        rhos, rs, al, be, ga, = DM_param
        ra, rho_s, rss, als, bes, gas = model_param
        G = 4.302*10**-6
        rhoss = 4*np.pi*G * rho_s

        phiDM = OMgenphi(r, rtrunc, rhos, rs, al, be, ga)
        phiL  = OMgenphi(r, rtrunc, rhoss, rss, als, bes, gas)

        return phiDM+phiL



# surface density of the alpha_beta_gamma stellar profile. Not normalized.
# integration is done using trapezoidal rule, for speed and stability.
def SigR_trapz(R, model_param, trapzN=1000, rlim=1):
        # Parameter:
        #      R [kpc]: projected radius the surface density
        #      model_param [list]: model parameters  
        #      trapzN: number of grid points to evaluate. The interval spacing is log-scaled.
        #      rlim: limiting radius (integration limit).
        # Return: surface density at R and limit in z coordinates. 
        ra, rho_s, rs_s, al_s, be_s, ga_s, = model_param

        if R>rlim:
                return 0, 1e-10

        def SigR_integrand(z):
                rr = z*z + R*R
                r  = rr**.5

                s = r/rs_s
                ww = z*z/(rs_s*rs_s)
                XX = R*R/(rs_s*rs_s)

                rhoL = rho_s*((ww+XX)**(-ga_s/2.)) * (1+(ww+XX)**(al_s/2.))**((ga_s-be_s)/al_s)
                return rhoL 

        zlim = ((rlim*rlim-R*R)**.5)*1
        #ans = 2 * integrate.quad(SigR_integrand, 0, zlim*1)[0]

        x = np.logspace(-10,np.log10(zlim), trapzN)
        y = SigR_integrand(x)
        ans = 2 * integrate.trapz(y, x)

        return ans, zlim

# Velocity Profile, P(vz|R), of the osipkov-merritt model integrated using trapzodial rule. 
# See Carollo et al. (1995) for the expression and details. 
def VP_trapz(R,vz, model_param, DM_param, GQ, r200, trapzN = 1000, rlim=1000):
        # Parameter:
        #      R [kpc]: projected radius the surface density
        #      vz [km/s]: line-of-sight velocity dispersion
        #      model_param [list]: model parameters. 
        #      DM_param [list]: dark matter density parameters.
        #      GQ [function]: funtion G(Q) of OM model, an output of the function
        #                     GQ(model_param, DM_param, num_rsteps, num_Qsteps), define below.
        #      r200: the radius at which enclosed average mass density is 200 times the critical
        #            density of the universe.
        #      trapzN: number of grid points to evaluate. The interval spacing is log-scaled.
        #      rlim: limiting radius (integration limit).
        # Return: surface density at R and limit in z coordinates.
        if R<0 or R>rlim:
                return 0.0

        rhos, rs, alpha, beta, gamma, = DM_param
        ra, rho_s, rs_s, al_s, be_s, ga_s, = model_param

        #r200 = getR200(DM_param)
        rtrunc = r200 # truncate the alpha_beta_gamma density at r200. 
        Plim = 0 # potential at infinity is 0 when the alpha_beta_gamma density is truncated 

        RR  = R*R
        rara= ra*ra

        def VP_integrand(z):
                rr = (z*z+R*R)
                r  = rr**.5

                #potential energy at radius r. 
                Phir = Plim - OMgenphi(r, rtrunc, rhos, rs, alpha, beta, gamma)
                #Phir = Plim - OMgenphi2(r, rtrunc, model_param, DM_param)

                g_rR = rara/((rara+rr)*(rara+z*z))**.5
                Qmax = Phir - 0.5*(rara+rr)*vz*vz/(rara+z*z)

                # integrant should return zero if Qmax<0
                Qmax_lt_0 = Qmax<0
                Qmax[Qmax_lt_0] = 0

                GQmax = GQ(Qmax)
                GQmax_lt_0 = GQmax < 0
                GQmax[GQmax_lt_0] = 0 #set it to zero if negative
                GQmax[Qmax_lt_0]  = 0 #Qmax lt zero, so it has to be zero

                integrand = g_rR * GQmax
                return integrand

        sigR, zlimR = SigR_trapz(R, model_param, trapzN=trapzN, rlim=rlim)
        x = np.logspace(-10,np.log10(zlimR), trapzN)
        y = VP_integrand(x)
        vp_int = integrate.trapz(y, x)

        ans = (2**.5/(np.pi*sigR)) * vp_int #/(2 *np.pi*np.pi * 2**.5)

        if sigR < 0: #surface density should be positive
                print 'sigR < 0', R, sigR
                print 'model_param = ', model_param
                print 'DM_param = ', DM_param, '\n'
                return 0
        if ans<0: # probability should be positive
                print 'NEG: Ri, vzi, sigR, VP = ', [R, vz, sigR, ans] #sigR
                print 'model_param = ', model_param
                print 'DM_param = ', DM_param, '\n'
                return 0

        return ans


# return stellar density normalization constant. 
def rhoLnorm(model_param):
        ra, rho_s, rs_s, al_s, be_s, ga_s, = model_param

        a = al_s*1.0
        b = be_s*1.0
        g = ga_s*1.0

        #works for a>0; b>3; g<3; && a is real..
        ans = ss.gamma((b-3)/a) * ss.gamma((3-g)/a) / ( a*ss.gamma((b-g)/a) )

        return 4 * rho_s * np.pi * ans * rs_s**3

#calculate r200, the radius in which enclosed mass density is 200 times of the critical density
#of the universe. assume stellar mass is negligible.
def getR200(DM_param):
        rhos, rs, al, be, ga, = DM_param

        G = 4.302*10**-6 #kpc/M_sun * (km/s)^2
        H = .072 #km/s / kpc
        rho_c = 3*H*H/(8*np.pi*G)

        #mass enclosed within in r
        def f2(r):
                x0 = 1e-10
                auxx = r/rs
                p1_a = ss.hyp2f1((3.-ga)/al, (be-ga)/al, (3+al-ga)/al, -x0**al)
                p1_b = ss.hyp2f1((3.-ga)/al,(be-ga)/al,(3.+al-ga)/al, -auxx**al)
                #note: rhos = 4Pi*G*rhos = ((21/.465)(1/rs))^2 = 4229.2 (km/s)^2 (1/kpc^2)
                Mr_DM = (rhos/G) * ( x0**(3.-ga) * p1_a - auxx**(3.-ga) * p1_b ) / ((ga - 3.))

                return Mr_DM*rs**3
	
        #get the index of the mass array that's closest to the critical mass Mc.
        rarr = np.linspace(rs, rs*10000, 1000000)
        Mc = (4*np.pi/3.0) * rarr**3 * rho_c * 200. #
        Mr = f2(rarr)
        min_indx = np.argmin(abs(Mr-Mc))

        #refine the search of r200.
        try:
                rarr2 = np.linspace(rarr[min_indx-5], rarr[min_indx+5], 1000)
                Mc2 = (4*np.pi/3.0) * rarr2**3 * rho_c * 200.
                Mr2 = f2(rarr2)
                min_indx2 = np.argmin(abs(Mr2-Mc2))
                r200 = rarr2[min_indx2]
        except:
                r200 = rarr[min_indx]

        return r200 * 1

#calculate r200, the radius in which enclosed mass density is 200 times of the critical density
#of the universe, include the stellar mass. Assumes alpha_beta_gamma stellar density.
def getR200b(model_param, DM_param):
        rhos, rs, al, be, ga, = DM_param
        ra, rho_s, rss, als, bes, gas = model_param

        G = 4.302*10**-6 #kpc/M_sun * (km/s)^2
        H = .072 #km/s / kpc; Hubble constant
        rho_c = 3*H*H/(8*np.pi*G)  #criticle density of the universe


	#mass enclosed within in r
        def f2(r):
                x0 = 1e-10
                auxx = r/rs
                p1_a = ss.hyp2f1((3.-ga)/al, (be-ga)/al, (3+al-ga)/al, -x0**al)
                p1_b = ss.hyp2f1((3.-ga)/al,(be-ga)/al,(3.+al-ga)/al, -auxx**al)
                #note: rhos = 4Pi*G*rhos; unit: (km/s)^2 * (1/kpc^2)
                Mr_DM = (rhos/G) * ( x0**(3.-ga) * p1_a - auxx**(3.-ga) * p1_b ) / ((ga - 3.))

                auxxs = r/rss
                pL_a = ss.hyp2f1((3.-gas)/als,(bes-gas)/als,(3.+als-gas)/als, -x0**als)
                pL_b = ss.hyp2f1((3.-gas)/als,(bes-gas)/als,(3.+als-gas)/als, -auxxs**als)
                #note: rho_s = rho_s ; unit: M_sun/kpc^3
                Mr_L = (rho_s*4*np.pi) * ( x0**(3.-gas) * pL_a - auxxs**(3.-gas) * pL_b ) / ((gas - 3.))

                return Mr_L*rss**3 + Mr_DM*rs**3

        #get the index of the mass array that's closest to the critical mass Mc.
        rarr = np.linspace(rs, rs*10000, 1000000)
        Mc = (4*np.pi/3.0) * rarr**3 * rho_c * 200. #
        Mr = f2(rarr)
        min_indx = np.argmin(abs(Mr-Mc))

        try:  #refine the search of r200.
                rarr2 = np.linspace(rarr[min_indx-5], rarr[min_indx+5], 1000)
                Mc2 = (4*np.pi/3.0) * rarr2**3 * rho_c * 200.
                Mr2 = f2(rarr2)
                min_indx2 = np.argmin(abs(Mr2-Mc2))
                r200 = rarr2[min_indx2]
        except:
                r200 = rarr[min_indx]

        return r200 * 1


# This function tabulate and get function G(Q) by first tabulate potential (Phi(r)) and stellar 
# density (rho(r)) to obtain rho(Phi), it is then integrated, for each Q to obtain G(Q). 
# The derivative of G(Q) is the density function f(Q). Note, f(Q) is not used to calculate P(vz|R), 
# but it is evaluated over the whole range of Q to check if the density function is physical.
def GQ(model_param, DM_param, num_rsteps = 1e5, num_Qsteps = 1000):
        # Parameters: 
        #         model_param [list]: model parameter
        #         DM_param [list]: dark matter density parameters
        #         num_rsteps: number of r-grid points to evalue to tabulate rhoQ(r) vs Phi(r).
        #         num_Qsteps: number of Q-grid points to evalue to tabulate G(Q).
        # return: G(Q): an interpolation funtion object.
        #         r200 [kpc] : r200 radius
        #         rmax [kpc] : limit of integration (use as infinity) 
        #
        # NOTE: we reject the density function as unphysical when f(Q) is negative over some % 
        # of the time. Check the condition on 'neg_Q_fraction' in the function for adjustment.
        rhos, rs, alpha, beta, gamma, = DM_param
        ra, rho_s, rs_s, al_s, be_s, ga_s, = model_param

        al_s = al_s*1.0
        be_s = be_s*1.0
        ga_s = ga_s*1.0

        r200   = getR200(DM_param)
        #r200   = getR200b(model_param, DM_param) #that's when considering the stellar mass contribution
        rmax   = r200*1000000 # use as r_infinity. 
        rtrunc = r200*1 #truncation radius where the mass density is set to zero beyond this radius.
        Plim = 0 # potential at infinity

        # To tabulate Phi(r) ----------------------------------------------------------------
        #---------------we want drho/dP over large range of rarr, so use rmax----------------
        t0 = time.time()
        # to obtain an array of radius to tabulate rho(r). (a full r range and logspacing 
        #                                                   might be sufficient.)
        rarr0 = np.linspace(1e-8, rmax*1, num_rsteps*.5)
        rarr1 = np.logspace(-5, np.log10(rmax)-0, num_rsteps*.5)
        rarr2 = np.logspace(-8, np.log10(rmax)-6, num_rsteps*.5)
        rarr = np.hstack((rarr0, rarr1, rarr2))
        rarr = np.unique(rarr)
        rarr = rarr[np.argsort(rarr)]

        Parr  = -1*(OMgenphi(rarr , rtrunc, rhos, rs, alpha, beta, gamma) )
        #Parr  = -1*(OMgenphi2(rarr , rtrunc, model_param, DM_param) ) # that's when considering the
                                                                       # stellar mass contribution


        # To tabulate rho_Q(r) ----------------------------------------------------------------
        #see Carollo et al. 1995 for rho_Q(r). Here we assume alpha_beta_gamma stellar density. 
        rhoQ = ( (1+rarr*rarr/(ra*ra))
                  * rho_s * ((rarr/rs_s)**-ga_s) * (1+(rarr/rs_s)**al_s)**((ga_s-be_s)/al_s) )


        #-----------------------------------------------------------------------------------------
        # --- Interpolate between rhoQ(r) and Phi(r) to obtain rhoQ(Phi) and its derivative ------
        rhoQ_sorted = rhoQ[np.argsort(Parr)]
        Parr_sorted = Parr[np.argsort(Parr)]
        Parr_sorted, Pindx = np.unique(Parr_sorted, return_index=True)
        rhoQ_sorted = rhoQ_sorted[Pindx]

        frhoQ  = PchipInterpolator(Parr_sorted, rhoQ_sorted, extrapolate=False)
        dfrhoQ = frhoQ.derivative()

        # Integrand of G(Q)
        def G_integrand(u, Q):
            phi = Q-u*u
            return -2 * dfrhoQ(phi)

        # Setup an array of Q to evaluate G(Q), and interpolate G(Q) function.
        rarrZ = np.logspace(-8, np.log10(r200)+1, int(num_Qsteps*.35))
        QarrZ  = -1*(OMgenphi(rarrZ , rtrunc, rhos, rs, alpha, beta, gamma) - Plim )
        # below is when considering the stellar mass contribution
        #QarrZ  = -1*(OMgenphi2(rarrZ , rtrunc, model_param, DM_param) - Plim)

        Qarr1 = np.linspace(0, max(Parr_sorted)*1, int(num_Qsteps*.65))
        Qarr = np.hstack((Qarr1, QarrZ))
        Qarr = np.unique(Qarr)
        Qarr = Qarr[np.argsort(Qarr)]

        Garr = [integrate.quad(G_integrand, Q**.5, 0, args=(Q,), full_output=1)[0]
                for Q in Qarr]

        Garr = np.nan_to_num( np.array(Garr) )

        # Interpolate Qarr and Garr to get G(Q) and its derivative f(Q)
        GQ = PchipInterpolator(Qarr, Garr, extrapolate=False)
        fQ = GQ.derivative()


        #-----------------------------------------------------------------------------------
        # Check if density function f(Q) is unphysical, if it's unphysica they density would
        # be interpreted as zero everywhere by setting r_infinity to negative
        numQ = 20000 # <-- you should adjust; number of Q to evaluate to check if f(Q)<0.
        Qtest = np.linspace(0, max(Qarr)*1, numQ)
        fQtest = fQ(Qtest)
        num_neg_fQ = sum(fQtest<0)
        neg_Q_fraction = num_neg_fQ / (numQ*1.)
        if neg_Q_fraction >= 0.001: # <-- you should adjust; the condition where we 
                                    # reject f(Q) as unphysical; currently set at 99.9%.
                rmax = -99

        return GQ, r200, rmax












