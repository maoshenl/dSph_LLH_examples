# dSph_LLH_examples
Methods of likelihood calculation or inference of dSph galaxy kinematic data.

The directories contains examples of likelihood calculation or inference using:

1. Osipkov-Merritt model (in ./om/)
    i) om_functions.py -- contain functions needed to calculate the conditional probability P(vz|R).
   ii) om_LLH.py --  examples of likelihood evaluation.

2. SFW model (in ./sfw/)
     i) sfw_functions.py -- contain functions to calculate probabilities of SFW(b=2) models.
    ii) sfw_ctype/sfw_function.cpp -- functions written using ctype to speed up the
                                      probability calculation.
   iii) sfw_LLH.py -- examples of likelihood evaluation.

3. likelihood approximation by KDE (in ./sfw_KDE-R/)
     i) sfw_kde_LLH.py -- an example of likelihood approximation using KDE.

4. Neural Network: (in ./sfw_nnet/)
     i) cvae.py -- an example of using Conditional Variational Autoencoder to infer SFW parameters.
    ii) mdn.py  -- an example of using Mixture Density Network to infer SFW parameters.
