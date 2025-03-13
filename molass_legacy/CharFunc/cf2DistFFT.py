"""
    SecTheory.cf2DistFFT.py

    Translated from MATLAB code by Viktor Witkovsky
    at https://github.com/witkovsky/CharFunTool/CF_InvAlgorithms/cf2DistFFT.m

    Copyright (c) 2022, SAXS Team, KEK-PF

%cf2DistFFT(cf,x,prob,options) Evaluates the CDF/PDF/QF (quantiles) from
% the characteristic function CF of a (continuous) distribution F by using
% the Fast Fourier Transform (FFT) algorithm. 
%
% The algorithm cf2DistFFT evaluates the approximate values CDF(x), PDF(x),
% and/or the quantiles QF(prob) for given x and prob, by interpolation from
% the PDF-estimate computed by the numerical inversion of the given
% characteristic function CF by using the FFT algorithm.
%
% SYNTAX:
%  result = cf2DistFFT(cf,x,prob,options)
%  [result,cdf,pdf,qf] = cf2DistFFT(cf,x,prob,options)
%
% INPUT:
%  cf      - function handle for the characteristic function CF,
%  x       - vector of values from domain of the distribution F, if x = [],
%            cf2DistFFT automatically selects vector x from the domain.
%  prob    - vector of values from [0,1] for which the quantiles will be
%            estimated, if prob = [], cf2DistFFT automatically selects
%            vector prob = [0.9,0.95,0.975,0.99,0.995,0.999].
%  options - structure with the following default parameters:   
%            options.isCompound = false  % treat the compound distributions
%                                        % of the RV Y = X_1 + ... + X_N,
%                                        % where N is discrete RV and X>=0 
%                                        % are iid RVs from nonnegative
%                                        % continuous distribution.
%             options.isInterp   = false % create and use the interpolant
%                                          functions for PDF/CDF/QF/RND
%             options.N = 2^10         % N points used by FFT
%             options.xMin = -Inf      % set the lower limit of X
%             options.xMax = Inf       % set the lower limit of X
%             options.xMean = []       % set the MEAN value of X
%             options.xStd = []        % set the STD value of X
%             options.dt = []          % set grid step dt = 2*pi/xRange
%             options.T = []           % set upper limit of (0,T), T = N*dt 
%             options.SixSigmaRule = 6 % set the rule for computing domain
%             options.tolDiff = 1e-4   % tol for numerical differentiation
%             options.isPlot = true    % plot the graphs of PDF/CDF
%  options.DIST - structure with information for future evaluations.
%                 options.DIST is created automatically after first call: 
%             options.DIST.xMin  = xMin   % the lower limit of X
%             options.DIST.xMax  = xMax   % the upper limit of X
%             options.DIST.xMean = xMean  % the MEAN value of X, 
%             options.DIST.cft   = cft    % CF evaluated at t_j : cf(t_j).   
%
% OUTPUTS:
%  result  - structure with the following results values:  
%             result.x                  = x;
%             result.cdf                = cdf;
%             result.pdf                = pdf;
%             result.prob               = prob;
%             result.qf                 = qf;
%             result.xFTT               = xFFT;
%             result.pdfFFT             = pdfFFT;
%             result.cdfFFT             = cdfFFT;
%             result.SixSigmaRule       = options.SixSigmaRule;
%             result.N                  = N;
%             result.dt                 = dt;
%             result.T                  = t(end);
%             result.PrecisionCrit      = PrecisionCrit;
%             result.myPrecisionCrit    = options.crit;
%             result.isPrecisionOK      = isPrecisionOK;
%             result.xMean              = xMean;
%             result.xStd               = xStd;
%             result.xMin               = xMin;
%             result.xMax               = xMax;
%             result.cf                 = cf;
%             result.options            = options;
%             result.tictoc             = toc;
%
% EXAMPLE 1:
%  % DISTRIBUTION OF A LINEAR COMBINATION OF THE INDEPENDENT RVs
%  % (Normal, Student's t, Rectangular, Triangular & Arcsine distribution)
%  % Y = X_{N} + X_{t} + 5*X_{R} + X_{T} + 10*X_{U}
%  % CFs: Normal, Student's t, Rectangular, Triangular, and Arcsine
%  cf_N  = @(t) exp(-t.^2/2);                                      
%  cf_t  = @(t,nu) min(1,besselk(nu/2, abs(t).*sqrt(nu),1) .* ...
%          exp(-abs(t).*sqrt(nu)) .* (sqrt(nu).*abs(t)).^(nu/2) / ...
%          2^(nu/2-1)/gamma(nu/2));     
%  cf_R  = @(t) min(1,sin(t)./t);   
%  cf_T  = @(t) min(1,(2-2*cos(t))./t.^2);
%  cf_U  = @(t) besselj(0,t);  
%  % Characteristic function of the linear combination Y
%  c    = [1 1 5 1 10]; nu = 1;
%  cf_Y = @(t) cf_N(c(1)*t) .* cf_t(c(2)*t,nu) .* cf_R(c(3)*t) .* ...
%         cf_T(c(4)*t) .* cf_U(c(5)*t);
%  clear options
%  options.N    = 2^10;
%  options.xMin = -50;
%  options.xMax = 50;
%  result = cf2DistFFT(cf_Y,[],[],options);
%  title('CDF of Y = X_{N} + X_{t} + 5*X_{R} + X_{T} + 10*X_{U}')
%
% EXAMPLE 2:
%  % DISTRIBUTION OF A LINEAR COMBINATION OF THE INDEPENDENT CHI2 RVs
%  % (Chi-squared RVs with 1 and 10 degrees of freedom)
%  % Y = 10*X_{Chi2_1} + X_{Chi2_10}
%  % Characteristic functions of X_{Chi2_1} and X_{Chi2_10}
%  %
%  df1       = 1;
%  df2       = 10;
%  cfChi2_1  = @(t) (1-2i*t).^(-df1/2);
%  cfChi2_10 = @(t) (1-2i*t).^(-df2/2);
%  cf_Y      = @(t) cfChi2_1(10*t) .* cfChi2_10(t);
%  clear options
%  options.xMin = 0;
%  result = cf2DistFFT(cf_Y,[],[],options);
%  title('CDF of Y = 10*X_{\chi^2_{1}} + X_{\chi^2_{10}}')
%
% EXAMPLE3 (PDF/CDF of the compound Poisson-Exponential distribution)
%  lambda1 = 10;
%  lambda2 = 5;
%  cfX  = @(t) cfX_Exponential(t,lambda2);
%  cf   = @(t) cfN_Poisson(t,lambda1,cfX);
%  x    = linspace(0,8,101);
%  prob = [0.9 0.95 0.99];
%  clear options
%  options.isCompound = 1;
%  result = cf2DistFFT(cf,x,prob,options)
%
% REMARKS:
%  The outputs of the algorithm cf2DistFFT are approximate values! The
%  precission of the presented results depends on several different
%  factors: 
%  - application of the FFT algorithm for numerical inversion of the CF
%  - selected number of points used by the FFT algorithm (by default
%    options.N = 2^10),
%  - estimated/calculated domain [A,B] of the distribution F, used with the
%    FFT algorithm. Optimally, [A,B] covers large part of the
%    distribution domain, say more than 99%. However, the default
%    automatic procedure for selection of the domain [A,B] may fail. It
%    is based on the 'SixSigmaRule': A = MEAN - SixSigmaRule * STD, and
%    B = MEAN + SixSigmaRule * STD. Alternatively, change the
%    options.SixSigmaRule to different value, say 12, or use the
%    options.xMin and options.xMax to set manually the values of A and B.
%
% REFERENCES:
% [1] WITKOVSKY, V.: On the exact computation of the density and of
%     the quantiles of linear combinations of t and F random
%     variables. Journal of Statistical Planning and Inference 94
%     (2001), 1-13.
% [2] WITKOVSKY, V.: Matlab algorithm TDIST: The distribution of a
%     linear combination of Student's t random variables. In COMPSTAT
%     2004 Symposium (2004), J. Antoch, Ed., Physica-Verlag/Springer
%     2004, Heidelberg, Germany, pp. 1995-2002.
% [3] WITKOVSKY, V.: WIMMER,G., DUBY, T. Logarithmic Lambert W x F
%     random variables for the family of chi-squared distributions
%     and their applications. Statistics & Probability Letters 96
%     (2015), 223-231.  
% [4] WITKOVSKY V. (2016). Numerical inversion of a characteristic
%     function: An alternative tool to form the probability distribution of
%     output quantity in linear measurement models. Acta IMEKO, 5(3), 32-44.
%
% SEE ALSO: cf2Dist, cf2DistGP, cf2DistGPT, cf2DistGPA, cf2DistFFT,
%           cf2DistBV, cf2CDF, cf2PDF, cf2QF 

% (c) Viktor Witkovsky (witkovsky@gmail.com)
% Ver.: 01-Sep-2020 13:25:21
%
% Revision history:
% Ver.: 15-Nov-2016 13:36:26
"""
import numpy as np

class Options:
    pass

class DistInfo:
    pass

def cf2DistFFT(cf, x=None, prob=None, options=Options()):

    if not hasattr(options, 'isCompound'):
        options.isCompound = False

    if not hasattr(options, 'N'):
        if options.isCompound:
            options.N = 2**14
        else:
            options.N = 2**10

    if not hasattr(options, 'xMin'):
        if options.isCompound:
            options.xMin = 0
        else:
            options.xMin = -np.inf

    if not hasattr(options, 'xMax'):
        options.xMax = np.inf

    if not hasattr(options, 'xMean'):
        options.xMean = None

    if not hasattr(options, 'xStd'):
        options.xStd = None

    if not hasattr(options, 'dt'):
        options.dt = None

    if not hasattr(options, 'T'):
        options.T = None

    if not hasattr(options, 'SixSigmaRule'):
        if options.isCompound:
            options.SixSigmaRule = 10
        else:
            options.SixSigmaRule = 6

    if not hasattr(options, 'tolDiff'):
        options.tolDiff = 1e-4

    if not hasattr(options, 'crit'):
        options.crit = 1e-12;

    if not hasattr(options, 'isPlot'):
        options.isPlot = True

    if not hasattr(options, 'DIST'):
        options.DIST = DistInfo()

    # Other options parameters
    if not hasattr(options, 'isPlotFFT'):
        options.isPlotFFT = False;

    if not hasattr(options, 'xN'):
        options.xN = 101;

    if not hasattr(options, 'chebyPts'):
        options.chebyPts = 2**9

    if not hasattr(options, 'isInterp'):
        options.isInterp = False

    ## GET/SET the DEFAULT parameters and the OPTIONS
    # First, set a special treatment if the real value of CF at infinity (large
    # value) is positive, i.e. const = real(cf(Inf)) > 0. In this case the
    # compound CDF has jump at 0 of size equal to this value, i.e. cdf(0) =
    # const, and pdf(0) = Inf. In order to simplify the calculations, here we
    # calculate PDF and CDF of a distribution given by transformed CF, i.e.
    # cf_new(t) = (cf(t)-const) / (1-const); which is converging to 0 at Inf,
    # i.e. cf_new(Inf) = 0. Using the transformed CF requires subsequent
    # recalculation of the computed CDF and PDF, in order to get the originaly
    # required values: Set pdf_original(0) =  Inf & pdf_original(x) =
    # pdf_new(x) * (1-const), for x > 0. Set cdf_original(x) =  const +
    # cdf_new(x) * (1-const).
    # 

    const = cf(1e30).real
    if options.isCompound:
        cfOld = cf
        if const > 1e-13:
            cf = lambda t: (cf(t) - const) / (1-const)

    if hasattr(options.DIST, 'xMean'):
        xMean              = options.DIST.xMean
        cft                = options.DIST.cft
        xMin               = options.DIST.xMin
        xMax               = options.DIST.xMax
        N                  = len(cft)
        k                  = np.arange(N)   # (0:(N-1))';
        xRange             = xMax - xMin
        dt                 = 2*np.pi / xRange
        t                  = (k - N/2 + 0.5) * dt
        xStd               = None
    else:
        N                  = 2*options.N
        dt                 = options.dt
        T                  = options.T
        xMin               = options.xMin
        xMax               = options.xMax
        xMean              = options.xMean
        xStd               = options.xStd
        SixSigmaRule       = options.SixSigmaRule
        tolDiff            = options.tolDiff
        cft                = cf(tolDiff*np.arange(1,5))     # (1:4)

        if xMean is None:
            xMean = ((-cft[1]
                + 8*cft[0]-8*cft[0].conj()
                + cft[1].conj())/(1j*12*tolDiff)).real

        if xStd is None:
            xM2 = (-(cft[3].conj()
                - 16*cft[2].conj()
                + 64*cft[1].conj()
                + 16*cft[0].conj()
                - 130 + 16*cft[0]
                + 64*cft[1]
                - 16*cft[2]+cft[3])/(144*tolDiff**2)).real
            xStd  = np.sqrt(xM2 - xMean**2)

        if np.isfinite(xMin) and np.isfinite(xMax):
            xRange = xMax - xMin
        elif T is not None:
            xRange = 2*np.pi / (2 * T / N)
            if np.isfinite(xMax):
                xMin = xMax - xRange
            elif np.isfinite(xMin):
                xMax = xMin + xRange
            else:
                xMin = xMean - xRange/2
                xMax = xMean + xRange/2
        elif dt is not None:
            xRange = 2*np.pi / dt
            if np.isfinite(xMax):
                xMin = xMax - xRange
            elif np.isfinite(xMin):
                xMax = xMin + xRange
            else:
                xMin = xMean - xRange/2
                xMax = xMean + xRange/2
        else:
            if np.isfinite(xMin):
                xMax = xMean + SixSigmaRule * xStd
            elif np.isfinite(xMax):
                xMin = xMean - SixSigmaRule * xStd
            else:
                xMin = xMean - SixSigmaRule * xStd
                xMax = xMean + SixSigmaRule * xStd
            xRange = xMax - xMin

        dt                 = 2*np.pi / xRange
        k                  = np.arange(N, dtype=complex)    # (0:(N-1))'
        t                  = (k - N/2 + 0.5) * dt
        cft                = cf(t[N//2:])
        cft                = np.concatenate([cft[::-1].conj(), cft])
        # cft[0]             = cft[0]/2
        # cft[N-1]           = cft[N-1]/2
        options.DIST.xMin  = xMin
        options.DIST.xMax  = xMax
        options.DIST.xMean = xMean
        options.DIST.cft   = cft

    ## ALGORITHM

    A      = xMin
    B      = xMax
    dx     = (B-A)/N
    c      = (-1)**(A*(N-1)/(B-A))/(B-A)
    print("A, B, N, dx, c=", A, B, N, dx, c)
    C      = c * (-1)**((1-1/N)*k)
    D      = (-1)**(-2*(A/(B-A))*k)     # k must be complex, see https://stackoverflow.com/questions/45384602/numpy-runtimewarning-invalid-value-encountered-in-power
    pdfFFT = np.max([np.zeros(N), (C*np.fft.fft(D*cft)).real], axis=0)

    # Reset the transformed CF, PDF, and CDF to the original values
    if options.isCompound:
        pdfFFT = pdfFFT * (1-const)
        pdfFFT[x==0] = np.inf

    return pdfFFT

if __name__ == '__main__':
    from scipy.special import jv, kve, gamma
    import matplotlib.pyplot as plt

    def ones(x):
        return 1 if np.isscalar(x) else np.ones(len(x))

    cf_N = lambda t: np.exp(-t**2/2)
    cf_t = lambda t, nu: np.min([ones(t), kve(nu/2, np.abs(t)*np.sqrt(nu)) * np.exp(-np.abs(t)* np.sqrt(nu)) * (np.sqrt(nu)*np.abs(t))**(nu/2) / 2**(nu/2-1) / gamma(nu/2)], axis=0)
    cf_R = lambda t: np.min([ones(t), np.sin(t)/t], axis=0)
    cf_T = lambda t: np.min([ones(t), (2-2*np.cos(t))/t**2], axis=0)
    cf_U = lambda t: jv(0, t)

    # Characteristic function of the linear combination Y
    c = np.array([1, 1, 5, 1, 10])
    nu = 1
    cf_Y = lambda t: cf_N(c[0]*t) * cf_t(c[1]*t,nu) * cf_R(c[2]*t) * cf_T(c[3]*t) * cf_U(c[4]*t)

    options = Options()
    options.N    = 2**10
    options.xMin = -50
    options.xMax = 50

    pdf = cf2DistFFT(cf_Y, None, None, options)

    print("pdf=", pdf[0:10])
    plt.plot(pdf)
    plt.show()
