import numpy as np
import matplotlib.pyplot as pl
import pickle
import pandas as pd
import copy
from contextlib import suppress
import os
import scipy.optimize
import scipy.stats
import time

class LC:
    """
    LC Class Object (LC==Light Curve)

    Takes in measurements of time, mag and magerr

    Parameters
    ----------
    x,y,yerr: lists or arrays
        time, mag and magerror

    file: str
        name of light curve file

    choices: dict
        from config.yaml file

    Returns
    ----------
    self.df : pandas DataFrame
        dataframe of x,y,yerr sorted by x from lowest to highest

    self.filename : str
        name of file

    self.choices : dict
        dictionary of analysis choices

    self.properties : dict
        maps self.filename to various aesthetic choices e.g. label and plot colour

    self.fourier_fits : dict
        empty dict used to initialise
    """

    def __init__(self,x,y,yerr,file,choices):
        x      = np.array([float(xi) for xi in x])
        y      = np.array([float(yi) for yi in y])
        yerr   = np.array([float(ye) for ye in yerr])

        df      = pd.DataFrame(data = dict(zip(['x','y','yerr'],[x,y,yerr])))
        df.sort_values('x',ascending=True,inplace=True)

        self.df           = df
        self.filename     = file
        self.choices      = choices
        self.properties   = dict(zip(self.choices['property_labels'],self.choices['files'][file]))
        self.fourier_fits = {}


    def fit_fourier_series_Nsinmax(self):
        '''
        Fit Fourier Series (Maximum Number of Sine Waves)

        Fits a Fourier series of the form Sum_{i=1}^{Nsinmax} A_i * sin(w_i * t + phi_i) to (x,y,yerr) data
        Performs fits sequentially, first fitting 1 sine wave,
        then using the best fit parameters as intialisations for fitting 2 sine waves,
        and so up to a maximum number of sine waves equal to Nsinmax==self.choices['Nsinmax']

        Returns
        ----------
        self.df.xp,self.df.yp,self.df.yperr : pandas Series in self.df
            defines pseudo data to be fitted, could be standardised if self.choices['standardise'] is True

        self.fourier_fits : dict
            key is Nsin (number of sine waves)
            values are dictionary:
            self explanatory, keys: ['initial_guess','best_params','deg_freedom','red_chisq','time','fit','paramerrs']
        '''

        standardise = self.choices['standardise']
        Nsinmax     = self.choices['Nsinmax']

        if standardise:#Easier to fit Fourier series when time is 0->1 and data is approx N(0,1)
            self.df['yp']    = (self.df.y-self.df.y.mean())/(self.df.y.std())
            self.df['yperr'] =  self.df.yerr/self.df.y.std()
            self.df['xp']    = (self.df.x-self.df.x.min())/(self.df.x.max()-self.df.x.min())
        else:
            self.df['yp']    = self.df.y
            self.df['yperr'] = self.df.yerr
            self.df['xp']    = self.df.x


        for Nsin in np.arange(1,Nsinmax+1,1):
            if not os.path.exists(f'products/{self.filename}_standardise{standardise}_Nsin{Nsin}.pkl'):
                #Function to create Fourier Series
                def fourier_series_model(x,vals):
                    function = vals[-1]
                    for i in range(len(vals)//3):
                        #function +=        Ai*   sin(         Bi*x+Ci         )
                        function  += vals[3*i]*np.sin(vals[3*i+1]*x+vals[3*i+2])
                    return function

                #Objective function to minimise is chi-squared
                def chisq(modelparams, x, y, yerr):
                    chisqval=0
                    for i in range(len(x)):
                        chisqval += ((y[i] - fourier_series_model(x[i], modelparams))/yerr[i])**2
                    return chisqval

                #If doing first fit, initialise parameter guesses with ones
                if Nsin==1:
                    initial_guess     = np.ones([3*Nsin+1])
                    initial_guess[-1] = self.df.yp.mean()#Last parameter is always the constant offset, therefore 3*Nsin+1 free parameters
                #Else, use previous best fitting parameters as initialisations
                else:
                    previous_best = self.fourier_fits[Nsin-1]['best_params']
                    initial_guess = np.concatenate(( previous_best[:-1], previous_best[-4:-1], np.array([previous_best[-1]]) ))

                deg_freedom  = self.df.shape[0] - initial_guess.size#v = N - n
                print ('###'*5)
                print (f'Beginning Fourier Series Fit using {Nsin} sine waves == {3*Nsin+1} params')
                time_start = time.clock()
                fit = scipy.optimize.minimize(chisq, initial_guess, args=(self.df.xp, self.df.yp, self.df.yperr))

                best_params = np.asarray(fit.x)
                paramerrs   = np.asarray([(fit.hess_inv[i][i])**0.5 for i in range(len(fit.x)) ])
                chisq       = chisq(best_params,self.df.xp, self.df.yp, self.df.yperr)
                red_chisq   = chisq/deg_freedom
                total_time  = time.clock()-time_start
                #Store best fitting parameters and save
                self.fourier_fits[Nsin] = dict(zip(['initial_guess','best_params','deg_freedom','red_chisq','time','fit','paramerrs'],[initial_guess,best_params,deg_freedom,red_chisq,total_time,fit,paramerrs]))
                print (fit)
                print (f'Achieved Reduced-Chi2 = {red_chisq:.3} using Nsin={Nsin} in {total_time:.3} seconds=={total_time/60:.3} minutes')
                with open(f'products/{self.filename}_standardise{standardise}_Nsin{Nsin}.pkl','wb') as f:
                    pickle.dump(self.fourier_fits[Nsin],f)
            else:
                with open(f'products/{self.filename}_standardise{standardise}_Nsin{Nsin}.pkl','rb') as f:
                    product = pickle.load(f)
                self.fourier_fits[Nsin] = product


    def get_optimum_Nsin(self):
        '''
        Get Optimum Number of Sine Waves

        Uses a list of reduced chi squared values to determine the best number of sine waves

        We set Occam's Razor at the level of dchi2v = 0.15s (i.e. any improvements less than this value are insignificant and fitting to noise)

        Returns
        ----------
        self.Nsin_opt : int
            optimum number of sine waves

        self.best_fourier_fit : dict
            fourier fit dictionary with Nsin=self.Nsin_opt
        '''
        chi2vs = {Nsin:self.fourier_fits[Nsin]['red_chisq'] for Nsin in self.fourier_fits}
        dchis  = np.asarray(list(chi2vs.values()))[:-1]-np.asarray(list(chi2vs.values()))[1:]
        for i,dchi in enumerate(dchis):
            if dchi<self.choices['max_dchi']:
                self.Nsin_opt = list(chi2vs.keys())[i]
                print (f'Optimum number of sine waves is {self.Nsin_opt}')
                break
            elif i==len(dchis)-1:
                raise Exception('Re-run with larger Nsinmax; reduced chi2 has not converged')
        self.best_fourier_fit = self.fourier_fits[self.Nsin_opt]

    def get_continous_fourier_function_and_errors(self, Nsin, input_t=None):
        '''
        Get Continuous Fourier Functino and Errors

        Uses a set of best fitting Fourier parameters, and computes a continuous function + errors

        Parameters
        ----------
        Nsin : int
            the number of sine waves to create Fourier series

        input_t : array or None (optional; default is None)
            if None, interpolate from minimum to maximum time of data

        Returns
        ----------
        self.tcont,self.mcont,self.mconterr : arrays
            corresponding to evenly spaced tgrid and interpolations using fitted Fourier series
        '''
        params    = self.fourier_fits[Nsin]['best_params']
        paramerrs = self.fourier_fits[Nsin]['paramerrs']

        if input_t is None:
            input_t  = np.linspace(self.df.xp.min(),self.df.xp.max(),self.choices['Ncont'])
        else:
            if self.choices['standardise']:
                input_t = (input_t-self.df.x.min())/(self.df.x.max()-self.df.x.min())

        function             = params[-1]
        function_err_squared = 0
        for i in range(Nsin):
            function += params[3*i]*np.sin(params[3*i+1]*input_t + params[3*i+2])
            #Using calculus and Gaussian approximation, errors on Fourier series interpolation
            function_err_squared += (            np.sin(params[3*i+1]*input_t + params[3*i+2]) * paramerrs[3*i]            )**2
            function_err_squared += (params[3*i]*np.cos(params[3*i+1]*input_t + params[3*i+2]) * paramerrs[3*i+1] * input_t)**2
            function_err_squared += (params[3*i]*np.cos(params[3*i+1]*input_t + params[3*i+2]) * paramerrs[3*i+2]          )**2

        function_err = function_err_squared**0.5

        if self.choices['standardise']:
            input_t      = input_t     *(self.df.x.max()-self.df.x.min()) + self.df.x.min()
            function     = function    *(self.df.y.std()) + self.df.y.mean()
            function_err = function_err*(self.df.y.std())

        self.tcont    = input_t
        self.mcont    = function
        self.mconterr = function_err



    def fourier_fit(self,Nsin='opt'):
        '''
        Fourier Fit

        Series of operations

        1) Fit Fourier series sequentially from Nsin=1 to Nsin=Nsinmax
        2) Determine optimum number of sine waves
        3) Use fit from optimum number of sine waves to get continuous interpolation on evenly spaced time grid

        Option to override optimum number of sine waves and choose Nsin = manual_input
        '''
        self.fit_fourier_series_Nsinmax()
        if Nsin=='opt':
            self.get_optimum_Nsin()
            Nsin = self.Nsin_opt
        self.get_continous_fourier_function_and_errors(Nsin)



    def plot(self, show=True, create_figure=True, invert=True, xlab="Time (MJD)", ylab="Magnitudes"):
        '''
        Plot

        Function used to plot light curves with various free options
        '''
        FS    = self.choices['FS']
        alpha = 0.25 ; capsize = 1 ; elinewidth = 0.5 ; marker = 'o' ; markersize = 2

        if create_figure:
            pl.figure()
            pl.title(f"{self.properties['band']} Light-Curve", fontsize=FS-3)

        lc         = self.df
        pl.errorbar(lc.x, lc.y, yerr = lc.yerr,
                        linestyle  = 'None',
                        marker     = marker,
                        markersize = markersize,
                        color      = self.properties['colour'],
                        ecolor     = self.properties['colour'],
                        capsize    = capsize,
                        elinewidth = elinewidth,
                        alpha      = alpha
                        )

        try:
            pl.plot(self.tcont,self.mcont,linestyle = '-',linewidth=2, color = 'black')
            pl.fill_between(self.tcont,self.mcont-self.mconterr,self.mcont+self.mconterr,alpha=0.2,color=self.properties['colour'])
        except Exception as e:
            pl.plot(lc.x, lc.y,  linestyle = '-', linewidth=1,  color = 'black', alpha=alpha,marker=None)

        if show:
            pl.xlabel(xlab, fontsize=FS)
            pl.ylabel(ylab, fontsize=FS)
            if invert:
                pl.gca().invert_yaxis()
            pl.tick_params(labelsize=FS)
            pl.tight_layout()
            pl.show()
