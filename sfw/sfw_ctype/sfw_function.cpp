#include <stdio.h>
#include <math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_hyperg.h>
#include <gsl/gsl_errno.h>
#include <vector>
#include <algorithm>    // std::sort
#include <stdlib.h>

#include <iostream>
#include <fstream>
#include <numeric>


/*********** begin extern "C" ****************/
extern "C" {

double genphi0( double x, double rhos, double rs, double al, double be, double ga);


struct params {
	double a, d, e, Ec, rlim, b, q, Jb, rhos, rs, al, be, ga;
};

struct vars {
	double x,y,z,vx,vy,vz, P0, Plim;
	params p1;
};

struct vars2 {
        double x,y,z, r, vx,vy,vz, P0, Plim, Pr;
        params p1;
};

struct vars3 {
	double x,y,z, r, R, vx,vy,vz, P0, Plim, Pr, C2, t, vesc;
	params p1;
};



/* express SFW model f(E,J) = f(x,y,z,vx,vy,vz) */
/*
 Parameter:
      x1 [kpc], x cartesian coordinates;
      y1 [kpc], y cartesian coordinates;
      z1 [kcp], z cartesian coordinates;
      vx [km/s], velocity in y-direction
      vy [km/s], velocity in x-direction 
      vz [km/s], line-of-sight velocity (z-direction)
      param, model parameters
      P0   [km/s]^2, potential energy at r=0, to be used as zero-point
      Plim [km/s]^2, potential energy at r=rlim, with P0 as the zero-point 
*/
double fXV( double x1, double y1, double z1, double vx, double vy, double vz, 
	double P0, double Plim, params p) {

        double Ps, Pr, J, E, xlim, N, gJ, hE,  Ec, Jb, x;
	double r, vi, vj, vk;

	vi = y1*vz-z1*vy;
        vj = -1*(x1*vz-z1*vx);
        vk = x1*vy-y1*vx;
	J = pow(vi*vi + vj*vj + vk*vk, .5);
	r = pow(x1*x1 + y1*y1 + z1*z1, .5);
	x = r/p.rs;

        Ps = p.rhos * p.rs * p.rs ;
        Pr = genphi0(x, p.rhos, p.rs, p.al, p.be, p.ga) - P0;
	E  = (vx*vx + vy*vy + vz*vz)/2.0 + Pr;


        Ec = p.Ec * Ps;
        xlim = p.rlim / p.rs;

        N = 1.0;
        if (E < Plim and E >= 0.) {
                hE = N * pow(E,p.a) * pow( pow(E,p.q)+pow(Ec,p.q), p.d/p.q ) * pow(Plim-E, p.e);
        } else {
                return 0.0;
        }


        Jb = p.Jb * p.rs * pow(Ps,.5);

        if (p.b <= 0){
           gJ = 1.0/(1 + pow(J/Jb,-p.b));
        } else {
           gJ = 1 + pow(J/Jb,p.b);
        }

        return  hE*gJ;
}


double fRV_intgd(double z, void *p) {
    vars &ps = *static_cast<vars *>(p);
    double fx = fXV( ps.x, ps.y, z, ps.vx, ps.vy, ps.vz, ps.P0, ps.Plim, ps.p1);
    return fx;
}


/* marginalize f(x,y,z,vx,vy,vz) over z, to obtain f(x,y,vx,vy,vz), 
 * to calculate the likelihood of 5D data */
double fRV( double x1, double y1, double vx, double vy, double vz,
        double P0, double Plim, params p) {
	
	double RR, zlim;
        RR = x1*x1+y1*y1;
	zlim = pow(p.rlim*p.rlim - RR, 0.5);

	double result, error;
	vars allparams;
	allparams.x = x1;
	allparams.y = y1;
	//allparams.z = z1;
	allparams.vx= vx;
	allparams.vy= vy;
	allparams.vz= vz;
	allparams.P0= P0;
	allparams.Plim = Plim;
	allparams.p1= p;	

	gsl_error_handler_t * old_handler=gsl_set_error_handler_off();

	gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000);
	gsl_function F;
	F.function = &fRV_intgd;
	F.params = static_cast<void *>(&allparams);
	gsl_integration_qags(&F, -zlim, zlim, 0, 1.49e-7, 1000, w, &result, &error);  
	gsl_integration_workspace_free (w);
	
        return 1.*result;
}


double hE(double E, double Plim, params p) {
	double Ec, Ps, hE, N;
	Ps = p.rhos * p.rs * p.rs;
	Ec = p.Ec * Ps;
	
	N = 1.0;
	if (E < Plim and E >= 0.) {
                hE = N * pow(E,p.a) * pow( pow(E,p.q)+pow(Ec,p.q), p.d/p.q ) * pow(Plim-E, p.e);
        } else {
                return 0.0;
        }

	return hE;
}

double gJ(double J, params p) {
	double Jb, Ps, gJ;

	Ps = p.rhos * p.rs * p.rs;
	Jb = p.Jb * p.rs * pow(Ps,.5);

        if (p.b <= 0){
           gJ = 1.0/(1 + pow(J/Jb,-p.b));
        } else {
           gJ = 1 + pow(J/Jb,p.b);
        }
	return gJ;
}

/* integrand for fXvzw, integrate over w*/
double fXvzw_intgd(double w, void *p) {
    double gw, E, hEaux, r,x,y,z,vx,vy,vz, c, s2; 
    double Jb, Ec, Ps, rs;

    vars2 &ps = *static_cast<vars2 *>(p);

    r  = ps.r;
    z  = ps.z;
    vz = ps.vz;
    c  = z/r;
    s2 = 1-c*c;
    rs = ps.p1.rs;
    Ps = ps.p1.rhos * rs * rs;
    Jb = ps.p1.Jb * rs * pow(Ps,0.5);
    gw = (2*s2*vz*vz + (1+c*c)*w*w) * M_PI; 
    E  = w*w/2. + vz*vz/2. + ps.Pr;
    hEaux = hE(E, ps.Plim, ps.p1);

    return hEaux * (2*M_PI + pow(r/Jb, ps.p1.b) * gw) * w;
}

/* Assuming b=2, calculate join probability P(x,y,z, vz) */
double fXvzw(double x1, double y1, double z1, double vz, 
		double P0, double Plim, params p) {

	double r, x, Pr, wlim, vesc1, error, result;

	r = pow(x1*x1 + y1*y1 + z1*z1, .5);
        x = r/p.rs;	
        Pr = genphi0(x, p.rhos, p.rs, p.al, p.be, p.ga) - P0;
	
	if (Plim > Pr){
	    vesc1 = pow(2*(Plim-Pr), 0.5);
	} else {
	    return 0.; }

	if (vesc1*vesc1-vz*vz > 0	){
	    wlim = pow( vesc1*vesc1-vz*vz, 0.5);
	} else {
	    return 0.; }

	vars2 allp2;
	allp2.x  = x1;
        allp2.y  = y1;
        allp2.z  = z1;	
	allp2.r  = r;
	allp2.vz = vz;
        allp2.P0 = P0;
        allp2.Plim = Plim;
	allp2.Pr = Pr;
        allp2.p1 = p;
	
	//turnoff the error handling so it won't abort and return the current best integration results
	gsl_error_handler_t * old_handler=gsl_set_error_handler_off(); 

	gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000);
        gsl_function F;
        F.function = &fXvzw_intgd;
        F.params = static_cast<void *>(&allp2);
        gsl_integration_qags(&F, 0, wlim, 1.0e-8, 1.00e-6, 1000, w, &result, &error);

        gsl_integration_workspace_free (w);

        return result;
} //end fXvzw


double fRvzw_intgd(double z, void *p){
	double ans;
	vars2 &ps = *static_cast<vars2 *>(p);
	
	ans = fXvzw(ps.x, ps.y, z, ps.vz, ps.P0, ps.Plim, ps.p1);
	return ans;
}

/* Assuming b=2, calculate join probability P(R, vz) */
double pRvz(double R, double vz, params p, double P0, double Plim){
	double x, y, zlim, error, result;
	x = R/pow(2., 0.5);
        y = x;
	if (p.rlim > R){
	    zlim = pow(p.rlim*p.rlim - R*R, 0.5);
	} else {return 0.;}

	vars2 allp2;
	allp2.x  = x;
        allp2.y  = y;
        allp2.vz = vz;
        allp2.P0 = P0;
        allp2.Plim = Plim;
        allp2.p1 = p;

	//turnoff the error handling so it won't abort
	gsl_error_handler_t * old_handler=gsl_set_error_handler_off();	

	gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000);
        gsl_function F;
        F.function = &fRvzw_intgd;
        F.params = static_cast<void *>(&allp2);
        gsl_integration_qags(&F, -zlim*0, zlim, 1.e-8, 1.00e-6, 1000, w, &result, &error);
        gsl_integration_workspace_free (w);

        return 2.*result;
}


// stellar density profile integrand, assuming b=2
double rhor_3_intgd(double v, void *p){
        double ans, E, v2;
        vars3 &ps = *static_cast<vars3 *>(p);

	v2 = v*v;
	E = v2*0.5 + ps.Pr;
        ans = (2 + ps.C2*pow(v,ps.p1.b)) * v2 * hE(E, ps.Plim, ps.p1);
        return ans;
}


// stellar density profile, assuming b=2
double rhor_3(double r, double P0, double Plim, params p) {
	double vesc, error, result, C2, Pr, x, Jb, Ps;
	vars3 allp3;
		
	Ps = p.rhos * p.rs*p.rs;
	Jb = p.Jb * p.rs * pow(Ps,0.5); //IMPORTANT..
	
	x = r/p.rs;
	Pr = genphi0(x, p.rhos, p.rs, p.al, p.be, p.ga) - P0;
	C2 = pow((r/Jb),p.b) * pow(M_PI, .5) * tgamma(1+0.5*p.b) / tgamma(0.5*(3+p.b));

	vesc = pow( (2.*(Plim - Pr)), .5 );

	allp3.p1 = p;
	allp3.Plim = Plim;
	allp3.Pr = Pr;
	allp3.C2 = C2;

	//turnoff the error handling so it won't abort
	gsl_error_handler_t * old_handler=gsl_set_error_handler_off();

        gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000);
        gsl_function F;
        F.function = &rhor_3_intgd;
        F.params = static_cast<void *>(&allp3);
        gsl_integration_qags(&F, 0, vesc, 1.00e-8, 1.00e-5, 1000, w, &result, &error);
        gsl_integration_workspace_free (w);

        return result*2*M_PI;
}


double pR_intgd(double z, void *p){
    double rho, r;   
    vars3 &ps = *static_cast<vars3 *>(p);

    r = pow(ps.R*ps.R + z*z, 0.5);
    if (r > ps.p1.rlim){
	return 0; }

    rho = rhor_3(r, ps.P0, ps.Plim, ps.p1);
    return rho;
}



/*# stellar surface density profile, assuming b=2, assuming b=2*/
double pR(double R, params p, double P0, double Plim){
    double vesc, error, result, C2, Pr;

    vars3 allp3;	
    allp3.R = R;
    allp3.P0 = P0;
    allp3.Plim = Plim;
    allp3.p1 = p;
   

    //turnoff the error handling so it won't abort
    gsl_error_handler_t * old_handler=gsl_set_error_handler_off();

    gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000);
    gsl_function F;
    F.function = &pR_intgd;
    F.params = static_cast<void *>(&allp3);
    gsl_integration_qags(&F, 0, p.rlim, 1.00e-8, 1.00e-5, 1000, w, &result, &error);
    gsl_integration_workspace_free (w);

    return 2*result; 
}



double norm_intgd(double r, void *p){
    double rho;
    vars3 &ps = *static_cast<vars3 *>(p);
    
    rho = rhor_3(r, ps.P0, ps.Plim, ps.p1);
    return rho*r*r;
}

/* # normalization constant of the model, assuming b=2 */
double norm(params p, double P0, double Plim){

    double error, result;

    vars3 allp3;
    allp3.P0 = P0;
    allp3.Plim = Plim;
    allp3.p1 = p;

    //turnoff the error handling so it won't abort
    gsl_error_handler_t * old_handler=gsl_set_error_handler_off();

    gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000);
    gsl_function F;
    F.function = &norm_intgd;
    F.params = static_cast<void *>(&allp3);
    gsl_integration_qags(&F, 0+1.e-8, p.rlim-1e-8, 1.e-8, 1.00e-5, 1000, w, &result, &error);
    gsl_integration_workspace_free (w);

    return 4*M_PI*result;
}



/*----------------------------------------------------------------------------------------------*/
/* Calculate SFW stellar density rhor (rhor0), surface density P(R) (pR0) and normalization 
 * constant (norm0), for arbitrary b values. ----------- */
/*----------------------------------------------------------------------------------------------*/
double rhor_aux_intgd(double v, void *p){
        double ans, E, v2, J;
        vars3 &ps = *static_cast<vars3 *>(p);

        v2 = v*v;
        E = v2*0.5 + ps.Pr;
        J = v*ps.r*sin(ps.t);

        ans = v2*gJ(J, ps.p1) * hE(E, ps.Plim, ps.p1);
        return ans;
}

/* integrate over velocity */
double rhor_aux(vars3 var){
        double vesc, error, result;

        vesc = var.vesc;  //pow( (2.*(var.Plim - var.Pr)), .5 );

        gsl_error_handler_t * old_handler=gsl_set_error_handler_off();

        gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000);
        gsl_function F;
        F.function = &rhor_aux_intgd;
        F.params = static_cast<void *>(&var);
        gsl_integration_qags(&F, 0, vesc, 1.00e-7, 1.00e-4, 1000, w, &result, &error);
        gsl_integration_workspace_free (w);

        return result;
}

/* stellar density integrand over angles t */
double rhor0_intgd(double t, void *p){

        vars3 &ps = *static_cast<vars3 *>(p);
        ps.t = t;

        return sin(t)*rhor_aux(ps);
}

/* stellar density profile, for arbitrary b */
double rhor0(vars3 var){
        double error, result;

        gsl_error_handler_t * old_handler=gsl_set_error_handler_off();

        gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000);
        gsl_function F;
        F.function = &rhor0_intgd;
        F.params = static_cast<void *>(&var);
        gsl_integration_qags(&F, 0, M_PI, 1.00e-7, 1.00e-4, 1000, w, &result, &error);
        gsl_integration_workspace_free (w);

        return 2*M_PI*result;
}

/* integrand: integrate over r */
double norm0_intgd(double r, void *p){
    double rho, x, Pr;
    vars3 &ps = *static_cast<vars3 *>(p);

    //r = pow(ps.R*ps.R + z*z, 0.5);
    x = r/ps.p1.rs;
    Pr = genphi0(x, ps.p1.rhos, ps.p1.rs, ps.p1.al, ps.p1.be, ps.p1.ga) - ps.P0;

    ps.r = r;
    ps.Pr = Pr;
    ps.vesc = pow( (2.*(ps.Plim - Pr)), .5 );

    if (r > ps.p1.rlim){
        return 0; }

    rho = rhor0(ps);
    return 4*M_PI*r*r*rho;
}

/* normalization constant for any b*/
double norm0(params p, double P0, double Plim){ 
    double vesc, error, result, Pr;

    vars3 allp3;
    //allp3.R = R;
    allp3.P0 = P0;
    allp3.Plim = Plim;
    allp3.p1 = p;

    /*
    if (R>=p.rlim){
        return 0;
    }*/

    /*turnoff the error handling so it won't abort */
    gsl_error_handler_t * old_handler=gsl_set_error_handler_off();

    gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000);
    gsl_function F;
    F.function = &norm0_intgd;
    F.params = static_cast<void *>(&allp3);
    gsl_integration_qags(&F, 0, p.rlim, 1.00e-7, 1.00e-4, 1000, w, &result, &error);
    gsl_integration_workspace_free (w);

    return result; 
}


/* integrand: marginalize stellar density over z */
double pR0_intgd(double z, void *p){
    double rho, x, r, Pr;
    vars3 &ps = *static_cast<vars3 *>(p);

    r = pow(ps.R*ps.R + z*z, 0.5);
    x = r/ps.p1.rs;
    Pr = genphi0(x, ps.p1.rhos, ps.p1.rs, ps.p1.al, ps.p1.be, ps.p1.ga) - ps.P0;
	
    ps.r = r;
    ps.Pr = Pr;
    ps.vesc = pow( (2.*(ps.Plim - Pr)), .5 ); 

    if (r > ps.p1.rlim){
        return 0; }

    rho = rhor0(ps);
    return rho;
}
	
/* surface density P(R) for arbitrary b */
double pR0(double R, params p, double P0, double Plim){
    double vesc, error, result, C2, Pr;

    vars3 allp3;
    allp3.R = R;
    allp3.P0 = P0;
    allp3.Plim = Plim;
    allp3.p1 = p;

    if (R>=p.rlim){
        return 0;
    }

    /*turnoff the error handling so it won't abort */
    gsl_error_handler_t * old_handler=gsl_set_error_handler_off();
    
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000);
    gsl_function F;
    F.function = &pR0_intgd;
    F.params = static_cast<void *>(&allp3);
    gsl_integration_qags(&F, 0, p.rlim, 1.00e-7, 1.00e-4, 1000, w, &result, &error);
    gsl_integration_workspace_free (w);
    
    return 2*result; 
}




/*************************************HYPERGEOMETRIC FUNCTION****************************************/
/* Hypergeometric function transformations to extend the range of z-value to calculate potential.
 See https://www.physicsforums.com/threads/calculating-hypergeometric-function-2f1-for-z-1.501956 
 for details */
double hyperg_z_LTN1 (double a, double b, double c, double z) {
    double coef1,coef2;

    coef1=tgamma(c)*tgamma(b-a)*pow(1-z,-a)/(tgamma(b)*tgamma(c-a));
    coef2=tgamma(c)*tgamma(a-b)*pow(1-z,-b)/(tgamma(a)*tgamma(c-b));
    /*printf ("coeff          = % .18f % .18f \n", tgamma(c-a), tgamma(b-a));
      printf ("g1          = % .18f \n", gsl_sf_hyperg_2F1(a,c-b,a-b+1,1./(1-z)) );
      printf ("g2          = % .18f \n", gsl_sf_hyperg_2F1(b,c-a,b-a+1,1./(1-z)) );*/
    return coef1*gsl_sf_hyperg_2F1(a,c-b,a-b+1,1./(1.-z))+coef2*gsl_sf_hyperg_2F1(b,c-a,b-a+1,1./(1.-z));
}

double hyperg_z_GT2 (double a, double b, double c, double z) {
    double coef1,coef2;

    coef1=tgamma(c)*tgamma(b-a)*pow(-z,-a)/(tgamma(b)*tgamma(c-a));
    coef2=tgamma(c)*tgamma(a-b)*pow(-z,-b)/(tgamma(a)*tgamma(c-b));
    /* printf ("coeff          = % .18f % .18f \n", coef1, coef2);
    printf ("g1          = % .18f \n", gsl_sf_hyperg_2F1(a,a-c+1,a-b+1,1./z));
    printf ("g2          = % .18f \n", gsl_sf_hyperg_2F1(b-c+1,b,b-a+1,1./z)); */
    return coef1*gsl_sf_hyperg_2F1(a,a-c+1,a-b+1,1./z)+coef2*gsl_sf_hyperg_2F1(b-c+1,b,b-a+1,1./z);
}

double hyperg_z_GT1LT2 (double a, double b, double c, double z) {
    double coef1,coef2;

    coef1 = tgamma(c)*tgamma(c-a-b)*pow(z,-a)/(tgamma(c-a)*tgamma(c-b));
    coef2 = tgamma(c)*tgamma(a+b-c)*pow(z, a-c)*pow(1-z,c-a-b)/(tgamma(a)*tgamma(b));
    /* printf ("coeff          = % .18f % .18f \n", coef1, coef2);
    printf ("g1          = % .18f \n", gsl_sf_hyperg_2F1(a,a-c+1,a+b-c+1,1-1./z));
    printf ("g2          = % .18f \n", gsl_sf_hyperg_2F1(c-a,1-a,c-a-b+1,1-1./z)); */
    return coef1*gsl_sf_hyperg_2F1(a,a-c+1,a+b-c+1,1-1./z)+coef2*gsl_sf_hyperg_2F1(c-a,1-a,c-a-b+1,1-1./z);
}

double hyp2f1(double a, double b, double c, double z){
    double result;
    if (fabs(z) < 1 ){
        //will give error: x=1, c - a - b = m for integer m.
        result = gsl_sf_hyperg_2F1(a,b,c,z);
    } else if (z <= -1){
	//error case 1(4.16): abs(b-a) = m for integer m.       
        result = hyperg_z_LTN1(a,b,c,z);
    } else if ((z > 1) && (z <= 2)) {
	//error case 5(4.19): abs(c - a - b) = m for integer m.  
        result = hyperg_z_GT1LT2(a,b,c,z);
    } else if (z > 2){
	//error case 6(4.20): abs(b-a) = m for integer m
        result = hyperg_z_GT2(a,b,c,z);
    }
    return result;
}
/********************************END HYPERGEOMETRIC FUNCTION******************************************/ 
/*
 General alpha_beta_gamma potential energy. Expression came from Mathematica by 
 integrating alpha_beta_gamma dark matter density. 
 REQUIRED: (gamma<3, beta>2, and alpha >0) */
double genphi0( double x, double rhos, double rs, double al, double be, double ga) {
        double x0, Ps, p1a, p1b, I1, p2, I2, ans1;
        x0 = pow(10,-13);
        Ps = rhos*rs*rs;

	//x0^(3-ga) is close to zero
        //p1a = hyp2f1((3.-ga)/al, (be-ga)/al, (3.+al-ga)/al, -pow(x0,al));
        p1b = hyp2f1((3.-ga)/al, (be-ga)/al, (3.+al-ga)/al, -pow(x, al));
        //I1  = (pow(x0,3-ga) * p1a - pow(x,3-ga) * p1b ) / (x * (ga - 3.));
	I1 = (0 - pow(x,3-ga) * p1b ) / (x * (ga - 3.));


        p2  = hyp2f1( (-2.+be)/al,(be-ga)/al,(-2.+al+be)/al, -pow(x,-al));
        I2  = pow(x, 2.-be) * p2 / (be -2.);

        ans1 = Ps * (1-(I1+I2));

        return ans1;
}


double genphi(double x, double rhos, double rs, double al, double be, double ga) {
        double x0 = pow(10,-13);
        double phi0 = 0; //genphi0(x0, rhos, rs, al, be, ga);

        //so it's zero at the center of the galaxy
        return genphi0(x, rhos, rs, al, be, ga) - phi0;
}



} //end extern "C"




