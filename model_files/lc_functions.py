from lc import *

def get_lc12_curves(lcs,choices,file):
    '''
    Get Curve of LC1-LC2 from two sets of data, using their Fourier series interpolation

    Firstly Determines common time grid;
    Then extracts optimum fourier parameters for each LC, and gets interpolation on common time grid
    Finally, compute LC1-LC2

    Parameters
    ----------
    lcs: dict
        keys are file names; values are LC class objects
    choices: dict
        analysis choices
    file: str
        name of new LC1-LC2 object

    Returns
    ----------
    lc1_common, lc2_common, lc12: class objects
        repsectively, the classes belonging to LC1 and LC2 but on a common time grid, and their difference
    '''
    lc1,lc2 = list(lcs.values())[:]

    min_t = min(lc1.df.x.min(),lc2.df.x.min())
    max_t = max(lc1.df.x.max(),lc2.df.x.max())

    common_t = np.linspace(min_t,max_t,choices['Ncont'])

    lc1_common = copy.deepcopy(lc1) ; lc2_common = copy.deepcopy(lc2)
    lc1_common.get_continous_fourier_function_and_errors(lc1_common.Nsin_opt, input_t=common_t)
    lc2_common.get_continous_fourier_function_and_errors(lc2_common.Nsin_opt, input_t=common_t)

    lc12m    = lc1_common.mcont - lc2_common.mcont
    lc12merr = ((lc1_common.mconterr)**2 + (lc2_common.mconterr)**2)**0.5
    lc12     = LC(common_t,lc12m,lc12merr,file,choices)
    return lc1_common, lc2_common, lc12

def get_Temperature(BV,choices):
    '''
    Get Temperature

    Simple function that uses Ballesteros 2012 to compute temperature from intrinsic B-V colour

    Parameters
    ----------
    BV : LC class object
        curve of apparent B-V colour
    choices: dict
        analysis choices

    Returns
    ----------
    T : LC class object
        curve of Temperature (Kelvin)
    '''
    BV.df.y += -choices['EBV']
    BVm    = BV.df.y
    BVmerr = BV.df.yerr
    Tm     = 4600*( 1 / ( 0.92*BVm + 1.7 ) +  1 / ( 0.92*BVm + 0.62 ) )
    Tmerr  = BVmerr*4600*0.92 * ( 1/ ((0.92*BVm+1.7)**2) + 1/ ((0.92*BVm + 0.62)**2) )
    T = LC(BV.df.x,Tm,Tmerr,'T_LC',choices)
    return T
