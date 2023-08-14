import numpy as np
from src.visualizations import qq_plot
from scipy import stats


# Mapping of continuous distribution names to scipy.stats objects
distribution_mapping_continuous = {
    'Normal': stats.norm,
    'Exponential': stats.expon,
    'Lognormal': stats.lognorm,
    'Gamma': stats.gamma,
    'Loggamma': stats.loggamma,
    'Beta': stats.beta,
    'Logistic': stats.logistic,
    'Gumbel_r': stats.gumbel_r,
    'Gumbel_l': stats.gumbel_l,  # gumbel, gumbel_l, and extreme1 are synonyms
    'Weibull': stats.weibull_min,
    'Pareto': stats.pareto,
    'Cauchy': stats.cauchy,
}


def continuous_fit_tests(data, distribution_name):

    # Get the distribution object from the mapping
    distribution = distribution_mapping_continuous[distribution_name]

    # Fit the distribution to the data
    params = distribution.fit(data)
    print(f"Best fit parameters for {distribution_name} Distribution: {params}")

    # Perform the Kolmogorov-Smirnov test
    D, p = stats.kstest(data, distribution.name, args=params)
    print(f"Kolmogorov-Smirnov test: D = {D}, p-value = {p}")

    # Perform the Anderson-Darling test
    if distribution.name in ['norm', 'expon', 'logistic', 'gumbel_l', 'gumbel_r', 'weibull_min']:
        A, critical_values, significance_levels = stats.anderson(data, distribution.name)
        print(f"Anderson-Darling test: A = {A}, Critical Values = {critical_values}, Significance Levels = {significance_levels}")
    else:
        print("Anderson-Darling test: Not supported for this distribution")

    # Perform the Cramér-von Mises test
    result = stats.cramervonmises(data, distribution.name, args=params)
    print(f"Cramér-von Mises test: W = {result.statistic}, p-value = {result.pvalue}")

    # Call the qq_plot function
    qq_plot(data, params, distribution, distribution_name)

    return params


distribution_mapping_discrete = {
    'Poisson': stats.poisson,
    'Geometric': stats.geom,
    'Negative Binomial': stats.nbinom,
}


def discrete_fit_tests(data, distribution_name):

    # Fit the distribution to the data
    distribution_name, params = fit_discrete_distributions(data, distribution_name)

    if len(params) == 1:
        print(f"Best fit parameter for {distribution_name} Distribution: {params[0]}")
    else:
        print(f"Best fit parameters for {distribution_name} Distribution: {params[0]}, {params[1]}")

    # Get the distribution object from the mapping
    distribution = distribution_mapping_discrete[distribution_name]

    # Compute the observed and expected frequencies for each unique value in the data
    unique_values, observed_frequencies = np.unique(data, return_counts=True)
    all_values = np.arange(np.min(unique_values), np.max(unique_values) + 1)
    expected_probabilities = distribution(*params).pmf(all_values)
    expected_frequencies = expected_probabilities * len(data)

    # Create a new version of observed_frequencies that includes zeros for the missing values
    observed_frequencies_extended = np.zeros_like(all_values)
    for value, frequency in zip(unique_values, observed_frequencies):
        observed_frequencies_extended[value - np.min(unique_values)] = frequency

    # Test only performs for the values present in the data
    mask = observed_frequencies_extended > 0
    observed_frequencies_masked = observed_frequencies_extended[mask]
    expected_frequencies_masked = expected_frequencies[mask]

    # Perform the G-test
    G = 2 * np.sum(observed_frequencies_masked * np.log(observed_frequencies_masked / expected_frequencies_masked))
    df = len(observed_frequencies_masked) - 1  # degrees of freedom
    p_G = stats.chi2.sf(G, df)  # The G statistic is chi-squared distributed with df degrees of freedom
    print(f"G-test: G = {G}, p-value = {p_G}")

    # Perform the chi-square test
    if np.all(expected_frequencies_masked >= 5):
        chi2, p_chi2 = stats.chisquare(observed_frequencies_masked, expected_frequencies_masked)
        print(f"Chi-square test: chi2 = {chi2}, p-value = {p_chi2}")
    else:
        print("Chi-square test: Not performed because some expected frequencies are less than 5")

    # Call the qq_plot function
    qq_plot(data, params, distribution, distribution_name)

    return params


def exAnte_logNormalDistribution_parameters(df):

    print("Lognormal | Poisson Parameters for each Scenario:")
    for index, row in df.iterrows():
        scenario = row['Scenario']
        L_typ = row['Typical Loss']
        L_ext = row['Extreme Loss']
        poisson_param = row['Yearly Frequency']

        sigma = 0.5*(stats.norm.ppf(0.99) + np.sqrt(stats.norm.ppf(0.99) ** 2 - 4*np.log(L_typ/L_ext)))
        mu = np.log(L_typ) + sigma ** 2

        print(f"{scenario}: mu={mu}, sigma={sigma}  |  Poisson: lambda={poisson_param}")


def fit_discrete_distributions(data, distribution_name):

    sample_mean = np.mean(data)
    sample_variance = np.var(data)

    if distribution_name == 'Poisson' or (distribution_name == 'Negative Binomial' and sample_variance <= sample_mean):
        # The parameter of the Poisson distribution is the mean of the data
        params = (sample_mean,)
    elif distribution_name == 'Geometric':
        # The parameter of the geometric distribution is 1 divided by the mean of the data
        params = (1 / sample_mean,)
    elif distribution_name == 'Negative Binomial' and sample_variance > sample_mean:
        # The parameters of the negative binomial distribution are estimated using method of moments
        n = sample_mean**2 / (sample_variance - sample_mean)  # n parameter of negative binomial
        p = n / (n + sample_mean)  # p parameter of negative binomial
        params = (n, p)

    if distribution_name == 'Negative Binomial' and sample_variance <= sample_mean:
        distribution_name = 'Poisson'  # switch to Poisson distribution
        print('\033[31m' + "Warning: Switched Negative Binomial to Poisson Distribution, since sample_variance <= sample_mean" + '\033[0m')

    return distribution_name, params
