import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf,acf

def main():  

  def get_distribution(Dataset):
    distribution = [ 
    '_binned_statistic', '_constants', '_continuous_distns', '_discrete_distns', '_distn_infrastructure', '_distr_params', '_multivariate', '_stats', '_stats_mstats_common', '_tukeylambda_stats', 'absolute_import', 'alpha', 'anderson', 'anderson_ksamp', 'anglit', 'ansari', 'arcsine', 'argus', 'bartlett', 'bayes_mvs', 'bernoulli', 'beta', 'betaprime', 'binned_statistic', 'binned_statistic_2d', 'binned_statistic_dd', 'binom', 'binom_test', 'boltzmann', 'boxcox', 'boxcox_llf', 'boxcox_normmax', 'boxcox_normplot', 'bradford', 'burr', 'burr12', 'cauchy', 'chi', 'chi2', 'chi2_contingency', 'chisquare', 'circmean', 'circstd', 'circvar', 'combine_pvalues', 'contingency', 'cosine', 'crystalball', 'cumfreq', 'describe', 'dgamma', 'dirichlet', 'distributions', 'division', 'dlaplace', 'dweibull', 'energy_distance', 'entropy', 'erlang', 'expon', 'exponnorm', 'exponpow', 'exponweib', 'f', 'f_oneway', 'fatiguelife', 'find_repeats', 'fisher_exact', 'fisk', 'fligner', 'foldcauchy', 'foldnorm', 'friedmanchisquare', 'gamma', 'gausshyper', 'gaussian_kde', 'genexpon', 'genextreme', 'gengamma', 'genhalflogistic', 'genlogistic', 'gennorm', 'genpareto', 'geom', 'gilbrat', 'gmean', 'gompertz', 'gumbel_l', 'gumbel_r', 'halfcauchy', 'halfgennorm', 'halflogistic', 'halfnorm', 'hmean', 'hypergeom', 'hypsecant', 'invgamma', 'invgauss', 'invweibull', 'invwishart', 'iqr', 'itemfreq', 'jarque_bera', 'johnsonsb', 'johnsonsu', 'kappa3', 'kappa4','ksone', 'kstat', 'kstatvar', 'kstest', 'kstwobign', 'kurtosis', 'kurtosistest', 'laplace', 'levene', 'levy', 'levy_l', 'levy_stable', 'linregress', 'loggamma', 'logistic', 'loglaplace', 'lognorm', 'logser', 'lomax', 'mannwhitneyu', 'matrix_normal', 'maxwell', 'mielke', 'mode', 'moment', 'mood', 'morestats', 'moyal', 'mstats', 'mstats_basic', 'mstats_extras', 'multinomial', 'multivariate_normal', 'mvn', 'mvsdist', 'nakagami', 'nbinom', 'ncf', 'nct', 'ncx2', 'norm', 'normaltest', 'norminvgauss', 'obrientransform', 'ortho_group', 'pareto', 'pearson3', 'pearsonr', 'percentileofscore', 'planck', 'pointbiserialr', 'poisson', 'power_divergence', 'powerlaw', 'powerlognorm', 'powernorm', 'ppcc_max', 'ppcc_plot', 'print_function', 'probplot', 'randint', 'random_correlation', 'rankdata', 'ranksums', 'rayleigh', 'rdist', 'recipinvgauss', 'reciprocal', 'relfreq', 'rice', 'rv_continuous', 'rv_discrete', 'rv_histogram', 'scoreatpercentile', 'sem', 'semicircular', 'shapiro', 'sigmaclip', 'skellam', 'skew', 'skewnorm', 'skewtest', 'spearmanr', 'special_ortho_group', 'statlib', 'stats', 't', 'test', 'theilslopes', 'tiecorrect', 'tmax', 'tmean', 'tmin', 'trapz', 'triang', 'trim1', 'trim_mean', 'trimboth', 'truncexpon', 'truncnorm', 'tsem', 'tstd', 'ttest_1samp', 'ttest_ind', 'ttest_ind_from_stats', 'ttest_rel', 'tukeylambda', 'tvar', 'uniform', 'unitary_group', 'variation', 'vonmises', 'vonmises_line', 'wald', 'wasserstein_distance', 'weibull_max', 'weibull_min', 'weightedtau', 'wilcoxon', 'wishart', 'wrapcauchy', 'zipf', 'zmap', 'zscore'
    ]
    distResults = []
    params = {}
    print()
    for distName in distribution:
        try:
            dist = getattr(stats, distName)
            param = dist.fit(Dataset)
            params[distName] = param
            D, p = stats.kstest(data, distName, args=param)
            print("P valor para: "+distName+" = "+str(p))
            distResults.append((distName, p))
        except Exception:
            pass

    print()
    bestDist, bestP = (max(distResults, key=lambda item: item[1]))
    return bestDist, bestP, params[bestDist]
 
  #load data set 
  data = pd.read_csv("data.txt",header=None)

  x= pd.DataFrame(np.array([x for x in range(1,24)]))

  y= pd.DataFrame(np.array(data))

  slope, intercept, r_value, p_value, std_err = stats.linregress(x[0],y[0])
  plt.plot(x, y, 'o', label='original data')
  
  plt.plot(x, intercept + slope*x, 'r', label='fitted line')

  plt.legend()

  plt.show()

  print("p valor es:",p_value)

  if(p_value > 0.05):
    print("La muestra es Identicamente distribuida: ID")

  print(acf(y[0]))
  plot_acf(y[0])
  plt.show()
  get_distribution(y[0])
  
if __name__ == '__main__':
     main()
     
    