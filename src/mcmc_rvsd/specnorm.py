import sys
import numpy as np
from uncertainties import unumpy

def input_chunck():
    args = sys.argv[1:]
    assert len(args) == 2, 'Please provide only 2 arguments'
    arg_start = int(args[0])
    arg_end = int(args[1])
    return arg_start, arg_end

def specNormg(order,spec,ivar_norm, niter=20, lowrej=1, highrej=3): # 归一化  （阶数、一维流量、流量误差）
    """
    Georges Kordopatis
    version: 29/03/2019
    INPUT:
    order: integer - order of the polynomial to fit the continuum
    niter: integer - number of iterations
    lowrej: float - a point is rejected when goes lowrej times sigma under the fit
    highrej: float - a point is rejected when goes highrej times sigma above the fit
    spec: fltarr - input spectrum
    OUTPUT: continuum at the same dimensions as spec and normalised spectrum
    """
   
    #order=1
   
    npix = len(spec)
    original = spec.copy()
    spectrum = spec.copy()
    x0 = np.arange(npix)

    for i in range(0, niter): # 迭代 去除吸收线的影响（使用上一次的拟合值）
        # fc=np.polyfit(x0, spectrum, order)
        # contin=np.poly1d(fc)
        seo = np.polynomial.polynomial.Polynomial.fit(x0, spectrum, order)
        contin = seo.linspace(n=npix)[1]

        residual = spectrum - contin
        sigma = np.nanstd(residual)  # 残差的标准差

        f = (spectrum < contin - lowrej * sigma) | (spectrum > contin + highrej * sigma)  # 判定异常
        spectrum[f] = contin[f]  # 替换异常为拟合值

    continuum = contin
    #normed = original / continuum
    
    
    err_flux_norm = 1 / np.sqrt(ivar_norm)   
            
            
    u = unumpy.uarray(original, err_flux_norm) # 整合成新的数组-原始数据
    v = unumpy.uarray(continuum, np.sqrt(continuum)) # 拟合数据 假设泊松分布
    y = u / v  # 归一化
    normed = unumpy.nominal_values(y)
    norm_err = unumpy.std_devs(y)
    
    #print(np.sqrt((original/continuum)**2*(((err_flux_norm/original)**2)+(np.sqrt(continuum)/continuum)**2)))
    return continuum, normed,norm_err

def vac2air(wave_vac):
    """
    Parameters
    ----------
    wave_vac:
        wavelength (A) in vacuum

    Return
    ------
    wave_air:
        wavelength (A) in air
    """
    wave_vac = np.array(wave_vac)
    s = 1e4 / wave_vac
    n = 1 + 0.0000834254 + \
        0.02406147 / (130 - s ** 2) + \
        0.00015998 / (38.9 - s ** 2)
    return wave_vac / n

def air2vac(wave_air):
    """
    Parameters
    ----------
    wave_air:
        wavelength (A) in air

    Return
    ------
    wave_vac:
        wavelength (A) in vacuum
    """
    wave_air = np.array(wave_air)
    s = 1e4 / wave_air
    n = 1 + 0.00008336624212083 + \
        0.02408926869968 / (130.1065924522 - s ** 2) + \
        0.0001599740894897 / (38.92568793293 - s ** 2)
    return wave_air * n

def shift_wave(wave, rv):
    """Shift the wavelength into the rest frame according to the radial velocity. Refer to
    https://github.com/tingyuansen/The_Payne/blob/e4bebc7eccb75c52dd0e7b419b5ab19b9a2f9df0/The_Payne/utils.py#L82

    Parameters
    ----------
    wave : array-like
        The wavelength array.
    rv : float
        The radial velocity.

    Returns
    -------
    rest_wave : array-like
        The rest wavelength array.
    interp_flux : array-like
        The interpolated flux array.
    """
    c = 299792.458  # the speed of light in km/s
    doppler_factor = np.sqrt((1 - rv / c) / (1 + rv / c))
    rest_wave = wave * doppler_factor
    return rest_wave

def shift_back_wave(rest_wave, rv):
    """Shift the rest wavelength back to the observer's frame according to the radial velocity.

    Parameters
    ----------
    rest_wave : array-like
        The rest wavelength array.
    rv : float
        The radial velocity.

    Returns
    -------
    observed_wave : array-like
        The observed wavelength array.
    """
    c = 299792.458  # the speed of light in km/s
    doppler_factor = np.sqrt((1 - rv / c) / (1 + rv / c))
    observed_wave = rest_wave / doppler_factor
    return observed_wave

def rebin_array(x, y, x_rebin):
    y_rebin = np.interp(x_rebin, x, y)
    return y_rebin

def cut_array(x, y, yerr, xmin, xmax):
    idx = np.where((x >= xmin) & (x <= xmax))
    x_cut = x[idx]
    y_cut = y[idx]
    yerr_cut = yerr[idx]

    return x_cut, y_cut, yerr_cut

def pre_detect(wave, flux_norm, snr, lower_lambda, upper_lambda):
    '''Pre-detection of the DIB 6283
    Parameters
    ----------
    wave : array-like
        The wavelength array
    flux_norm : array-like
        The normalized flux array
    snr : float
        The signal-to-noise ratio
    lower_lambda : float
        The lower detection boundary of the wavelength range
    upper_lambda : float
        The upper detection boundary of the wavelength range

    Returns
    -------
    detect_flag : int
        0 or 1, 0 for undetected, 1 for detected
    wave_c : float
        the central wavelength of the DIB
    a_v : float
        the absorption depth of the DIB
    threshold : float
        the threshold of the absorption depth
    '''
    t = (wave >= lower_lambda) & (wave <= upper_lambda)
    r = flux_norm[t]
    w = wave[t]

    # DISCARDED METHOD
    # i = np.argmin(r)
    # if i-2 < 0:
    #     lowy = 1. - np.nanmean(r[i:i+5])
    #     lowx = np.nanmean(w[i:i+5])
    # elif i+3 > r.shape[0]:
    #     lowy = 1. - np.nanmean(r[i-4:i+1])
    #     lowx = np.nanmean(w[i-4:i+1])
    # else:
    #     lowy = 1. - np.nanmean(r[i-2:i+3])
    #     lowx = np.nanmean(w[i-2:i+3])

    lowy = 1. - np.min(r)
    lowx = w[np.argmin(r)]

    sdv = 1. / snr
    threshold = 1 * sdv  # 3 sigma which can be modified if needed

    a_v = np.abs(lowy)
    wave_c = lowx

    if a_v >= threshold:
        detect_flag = 1
    else:
        detect_flag = 0
    return detect_flag, wave_c, a_v, threshold

def dynamic_mask(wave, flux, cw, width):

    flux_copy = flux.copy()  # ! NOTE: The original flux array will be modified if a copy is not made
    # if type(cw) != list:
    if not isinstance(cw, list):
        lower_w = cw - width
        upper_w = cw + width
    else:
        lower_w = cw[0] - width
        upper_w = cw[1] + width
    mask = (wave <= lower_w) | (wave >= upper_w)
    flux_copy[mask] = 1.

    return flux_copy
    
def dynamic_mask_copy(wave, flux, lower_w, upper_w):
    flux_copy = flux.copy()
    mask = (wave <= lower_w) | (wave >= upper_w)
    flux_copy[mask] = 1.

    return flux_copy    



def dynamic_mask_inte(wave, flux, cw, width):
    ''' Dynamically determine the range of mask

    Parameters
    ----------
    wave : array-like
        The wavelength array
    flux : array-like
        The flux array
    cw : float or list-like
        The central wavelength of the DIB
    width : float
        The width needs to remain

    Returns
    -------
    flux_copy : array-like
        The masked flux array
    '''
    flux_copy = flux.copy()  # ! NOTE: The original flux array will be modified if a copy is not made
    # if type(cw) != list:
    if not isinstance(cw, list):
        lower_w = cw - width
        upper_w = cw + width
    else:
        lower_w = cw[0] - width
        upper_w = cw[1] + width
    

    return lower_w,upper_w

def gaussian(x, a, b, c):
    return a * np.exp(-(x - b) ** 2 / (2 * c ** 2))


def gaussian_c(x, a, b, c, d):
    # Here, `d` is the continuum level, the default value is fixed to 1.0
    return gaussian(x, a, b, c) + d


# =========================================================================
# Define the log probability function for the MCMC
def pr_a(a, a0):
    width = 0.5 * np.abs(a0)
    return np.exp(-(a - a0) ** 2 / (2.0 * width ** 2))


def pr_mu(mu, mu0):
    width = 0.5
    return np.exp(-(mu - mu0) ** 2 / (2.0 * width ** 2))


def pr_sigma(sigma, sigma0):
    # sigma - measured width for Gaussian profile
    width = 0.5
    return np.exp(-(sigma - sigma0) ** 2 / (2.0 * width ** 2))

def pr_d(d, d0):
    # sigma - measured width for Gaussian profile
    width = 0.5
    return 1

def lnlike_6614(p, x, y, yerr):
    inv_sigma2 = 1.0 / (yerr**2)
    return -0.5 * (np.sum((y - gaussian_c(x, *p)) ** 2 * inv_sigma2 - np.log(inv_sigma2)))


def lnprior_6614(p, p_guess):
    # unpack the parameters:
    a_6614, mu_6614, sigma_6614,d_6614 = p
    a0_6614, mu0_6614, sigma0_6614,d0_6614 = p_guess

    flag_a_6614 = (a_6614 >= -0.4) & (a_6614 < 0.0)
    flag_mu_6614 = (mu_6614 >= 6608.0) & (mu_6614 < 6620.0)
    flag_sigma_6614 = (sigma_6614 >= 0.1) & (sigma_6614 < 5)
    flag_d_6614=(d_6614>=0.5) & (d_6614<=1.5)

    # Gaussian priors:
    if flag_a_6614 & flag_mu_6614 & flag_sigma_6614 & flag_d_6614:
        lp_a_6614 = np.log(pr_a(a_6614, a0_6614))
        # lp_a_6614 = 0.0
        lp_mu_6614 = np.log(pr_mu(mu_6614, mu0_6614))
        lp_sigma_6614 = np.log(pr_sigma(sigma_6614, sigma0_6614))
        lp_d_6614=np.log(pr_d(d_6614,d0_6614))
        return lp_a_6614 + lp_mu_6614 + lp_sigma_6614+lp_d_6614

    # Uniform priors:
    # if flag_a_6614 & flag_mu_6614 & flag_sigma_6614:
    #     return 0.0

    return -np.inf


def lnprob_6614(p, p_guess, x, y, yerr):
    lp = lnprior_6614(p, p_guess)
    if not np.isfinite(lp):
        return -np.inf
    total = lp + lnlike_6614(p, x, y, yerr)
    if np.isnan(total):
        return -np.inf
    return total

def get_6614_cf(wave, flux_mask, mu0, a0, sigma0=0.5):
    ''' Measure the DIB 6614 using the 'C'urve 'F'itting method
    Parameters
    ----------
    wave : array
        The wavelength of the spectrum which has been preprocessed and processed
    flux_mask : array
        The masked flux
    mu0 : float
        The initial guess of the central wavelength of the DIB 6614 after pre-detection
    a0 : float
        The initial guess of the absorption depth of the DIB 6614 after pre-detection
    sigma0 : float, Default = 2.0
        The initial guess of the standard deviation of the DIB 6614 after pre-detection

    Returns
    -------
    params : array
        The fitted parameters of the DIB 6614 gaussian profile
        params[0] : The absorption depth
        params[1] : The central wavelength
        params[2] : The standard deviation
    perr : array
        The error of the fitted parameters of the DIB 6614 gaussian profile, i.e., the standard deviation of the `params`
    '''
    initial_guess = [-a0, mu0, sigma0,0.98]
    bounds = ([-0.4, 6611.5, 0.1,0.5],
              [0.0, 6617, 5.0,1.5])
    params, pcov = curve_fit(gaussian_c, wave, flux_mask, p0=initial_guess, bounds=bounds, maxfev=100000)
    perr = np.sqrt(np.diag(pcov))

    return np.array(params), np.array(perr)

def get_6614_mcmc(wave, flux_mask, params, flux_err):
    '''
    Parameters
    ----------
    wave : array
        The wavelength of the spectrum which has been preprocessed and processed
    flux_mask : array
        The masked flux
    params : array
        The DIB 6614 parameters obtained from the curve fitting method
    flux_err : array, Default = 0.0
        The error of the flux. Actually, it is the inverse variance = 1 / error^2

    Returns
    -------
    pfit : array
        The fitted parameters of the DIB 6614 gaussian profile
    perr : array
        The error of the fitted parameters of the DIB 6614 gaussian profile
        The error is calculated as the mean value of the 16th and 84th percentiles of the samples
    samples : array
        The samples of the fitted parameters of the DIB 6614 gaussian profile
    '''
    nwalkers = 100
    ndim = params.shape[0]
    if np.sum(flux_err) == 0.0:
        flux_err = 1e-3 * np.ones_like(flux_mask)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_6614, args=(params, wave, flux_mask, flux_err))
    pos = params + np.random.normal(0, 0.01, (nwalkers, ndim))
 
    # Do the burn-in
    s1, _, _ = sampler.run_mcmc(pos, 50)
    # Do the production
    sampler.reset()
    sampler.run_mcmc(s1, 200)
    samples = sampler.flatchain
    results = np.percentile(samples, [16, 50, 84], axis=0)
    pfit = results[1]
    perr = np.abs(results[0] - results[2]) / 2.0

    return pfit, perr, samples


def lnlike_6610(p, x, y, yerr):
    inv_sigma2 = 1.0 / (yerr**2)
    return -0.5 * (np.sum((y - gaussian_c(x, *p)) ** 2 * inv_sigma2 - np.log(inv_sigma2)))


def lnprior_6610(p, p_guess):
    # unpack the parameters:
    a_6614, mu_6614, sigma_6614,d_6614 = p
    a0_6614, mu0_6614, sigma0_6614,d0_6614 = p_guess

    flag_a_6614 = (a_6614 >= -0.4) & (a_6614 < 0.0)
    flag_mu_6614 = (mu_6614 >= 6606.0) & (mu_6614 < 6612.0)
    flag_sigma_6614 = (sigma_6614 >= 0.1) & (sigma_6614 < 5)
    flag_d_6614=(d_6614>=0.5) & (d_6614<=1.5)

    # Gaussian priors:
    if flag_a_6614 & flag_mu_6614 & flag_sigma_6614 & flag_d_6614:
        lp_a_6614 = np.log(pr_a(a_6614, a0_6614))
        # lp_a_6614 = 0.0
        lp_mu_6614 = np.log(pr_mu(mu_6614, mu0_6614))
        lp_sigma_6614 = np.log(pr_sigma(sigma_6614, sigma0_6614))
        lp_d_6614=np.log(pr_d(d_6614,d0_6614))
        return lp_a_6614 + lp_mu_6614 + lp_sigma_6614+lp_d_6614

    # Uniform priors:
    # if flag_a_6614 & flag_mu_6614 & flag_sigma_6614:
    #     return 0.0

    return -np.inf


def lnprob_6610(p, p_guess, x, y, yerr):
    lp = lnprior_6610(p, p_guess)
    if not np.isfinite(lp):
        return -np.inf
    total = lp + lnlike_6610(p, x, y, yerr)
    if np.isnan(total):
        return -np.inf
    return total

def get_6610_cf(wave, flux_mask, mu0, a0, sigma0=0.5):
    ''' Measure the DIB 6614 using the 'C'urve 'F'itting method
    Parameters
    ----------
    wave : array
        The wavelength of the spectrum which has been preprocessed and processed
    flux_mask : array
        The masked flux
    mu0 : float
        The initial guess of the central wavelength of the DIB 6614 after pre-detection
    a0 : float
        The initial guess of the absorption depth of the DIB 6614 after pre-detection
    sigma0 : float, Default = 2.0
        The initial guess of the standard deviation of the DIB 6614 after pre-detection

    Returns
    -------
    params : array
        The fitted parameters of the DIB 6614 gaussian profile
        params[0] : The absorption depth
        params[1] : The central wavelength
        params[2] : The standard deviation
    perr : array
        The error of the fitted parameters of the DIB 6614 gaussian profile, i.e., the standard deviation of the `params`
    '''
    initial_guess = [-a0, mu0, sigma0,1]
    '''bounds = ([-0.4, 6611, 0.1,0.5],
              [0.0, 6617, 5.0,1.5])'''
    bounds = ([-0.5, 6606, 0.1,0.5],
              [0.0, 6612, 5.0,1.5])   
    params, pcov = curve_fit(gaussian_c, wave, flux_mask, p0=initial_guess, bounds=bounds, maxfev=100000)
    perr = np.sqrt(np.diag(pcov))

    return np.array(params), np.array(perr)

def get_6610_mcmc(wave, flux_mask, params, flux_err=0.0):
    '''
    Parameters
    ----------
    wave : array
        The wavelength of the spectrum which has been preprocessed and processed
    flux_mask : array
        The masked flux
    params : array
        The DIB 6614 parameters obtained from the curve fitting method
    flux_err : array, Default = 0.0
        The error of the flux. Actually, it is the inverse variance = 1 / error^2

    Returns
    -------
    pfit : array
        The fitted parameters of the DIB 6614 gaussian profile
    perr : array
        The error of the fitted parameters of the DIB 6614 gaussian profile
        The error is calculated as the mean value of the 16th and 84th percentiles of the samples
    samples : array
        The samples of the fitted parameters of the DIB 6614 gaussian profile
    '''
    nwalkers = 100
    ndim = params.shape[0]

    if np.sum(flux_err) == 0.0:
        flux_err = 1e-3 * np.ones_like(flux_mask)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_6610, args=(params, wave, flux_mask, flux_err))
    pos = params + np.random.normal(0, 0.01, (nwalkers, ndim))
    # Do the burn-in
    s1, _, _ = sampler.run_mcmc(pos, 50)
    # Do the production
    sampler.reset()
    sampler.run_mcmc(s1, 200)
    samples = sampler.flatchain
    results = np.percentile(samples, [16, 50, 84], axis=0)
    pfit = results[1]
    perr = np.abs(results[0] - results[2]) / 2.0

    return pfit, perr, samples


def lnlike_6633(p, x, y, yerr):
    inv_sigma2 = 1.0 / (yerr**2)
    return -0.5 * (np.sum((y - gaussian_c(x, *p)) ** 2 * inv_sigma2 - np.log(inv_sigma2)))


def lnprior_6633(p, p_guess):
    # unpack the parameters:
    a_6614, mu_6614, sigma_6614,d_6614 = p
    a0_6614, mu0_6614, sigma0_6614,d0_6614 = p_guess

    flag_a_6614 = (a_6614 >= -0.5) & (a_6614 < 0.0)
    flag_mu_6614 = (mu_6614 >= 6632) & (mu_6614 < 6636)
    flag_sigma_6614 = (sigma_6614 >= 0) & (sigma_6614 < 5)
    flag_d_6614=(d_6614>=0.5) & (d_6614<=1.5)

    # Gaussian priors:
    if flag_a_6614 & flag_mu_6614 & flag_sigma_6614 & flag_d_6614:
        lp_a_6614 = np.log(pr_a(a_6614, a0_6614))
        # lp_a_6614 = 0.0
        lp_mu_6614 = np.log(pr_mu(mu_6614, mu0_6614))
        lp_sigma_6614 = np.log(pr_sigma(sigma_6614, sigma0_6614))
        lp_d_6614=np.log(pr_d(d_6614,d0_6614))
        return lp_a_6614 + lp_mu_6614 + lp_sigma_6614+lp_d_6614

    # Uniform priors:
    # if flag_a_6614 & flag_mu_6614 & flag_sigma_6614:
    #     return 0.0

    return -np.inf


def lnprob_6633(p, p_guess, x, y, yerr):
    lp = lnprior_6633(p, p_guess)
    if not np.isfinite(lp):
        return -np.inf
    total = lp + lnlike_6633(p, x, y, yerr)
    if np.isnan(total):
        return -np.inf
    return total

def get_6633_cf(wave, flux_mask, mu0, a0, sigma0=0.5):
    
    initial_guess = [-a0, mu0, sigma0,1]
    bounds = ([-0.5, 6631, 0.1,0.5],
              [0.0, 6636, 5.0,1.5])   
    params, pcov = curve_fit(gaussian_c, wave, flux_mask, p0=initial_guess, bounds=bounds, maxfev=100000)
    perr = np.sqrt(np.diag(pcov))

    return np.array(params), np.array(perr)

def get_6633_mcmc(wave, flux_mask, params, flux_err=0.0):
   
    nwalkers = 100
    ndim = params.shape[0]

    if np.sum(flux_err) == 0.0:
        flux_err = 1e-3 * np.ones_like(flux_mask)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_6633, args=(params, wave, flux_mask, flux_err))
    pos = params + np.random.normal(0, 0.01, (nwalkers, ndim))
    # Do the burn-in
    s1, _, _ = sampler.run_mcmc(pos, 50)
    # Do the production
    sampler.reset()
    sampler.run_mcmc(s1, 200)
    samples = sampler.flatchain
    results = np.percentile(samples, [16, 50, 84], axis=0)
    pfit = results[1]
    perr = np.abs(results[0] - results[2]) / 2.0

    return pfit, perr, samples




def ew_measure(depth, depth_err, sigma, sigma_err):
    
    # ew_err = np.sqrt((np.sqrt(2.0*np.pi) * sigma * depth_err)**2 + (np.sqrt(2.0*np.pi) * depth * sigma_err)**2)
    u = ufloat(depth, depth_err)
    v = ufloat(sigma, sigma_err)
    y = np.abs(np.sqrt(2.0 * np.pi) * u * v)
    ew = y.n
    ew_err = y.s
    return ew, ew_err

def compute_int_ew_dib(wave, flux, d,lower_lambda, upper_lambda):
    idx = np.where((wave >= lower_lambda) & (wave <= upper_lambda))
    int_ew = np.trapz(np.abs(d - flux[idx]), wave[idx]) # changed to absolute value
    return int_ew,wave[idx],flux[idx]