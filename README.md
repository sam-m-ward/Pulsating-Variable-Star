# Pulsating-Variable-Star
Fourier series fitting using $\chi^2$-minimisation

This notebook fits B and V light curves of a pulsating variable star: AH Cam

### Method

It uses scipy.optimise to fit a Fourier series of the form:

$f(t) = A_0 + \sum_{i=1} A_i sin(\omega_i t + \phi_i)$,

where maximum $i=N_{sin}$. Fitting is performed sequentially, first fitting with $N_{sin}=1$, then using these best-fit parameters as intialisations for $N_{sin}=2$, and so on... 

Optimum number of sine waves is determined by where $\Delta \chi^2_{\nu}<0.15$, as $N_{sin}$ is increased by 1.

Algorithm standardises data, ensuring time is in (0,1), and brightness is approximately distributed as a standard normal. Code applies inverse transformation once best Fourier parameters have been found.

Finally, Fourier interpolations for both B and V are put on a common time grid to compute a B-V Fourier series, from which Temperature can be derived.

### Example Use Case

See [**fit_fourier_series.ipynb**](https://github.com/sam-m-ward/Pulsating-Variable-Star/blob/main/fit_fourier_series.ipynb).

These techniques were used in [**BScReport.pdf**](https://github.com/sam-m-ward/Analysis_Reports/blob/main/BScReport.pdf)
