import matplotlib.pyplot as plt
import numpy as np


# Plot the distribution of the number of damages per Year
def plot_lossesPerYear(timeStampSeries):
    timeStampSeries.plot(kind='hist', rwidth=0.8, title='Distribution of Losses per Year', density=True)
    plt.xlabel('Number of Losses per Year')
    plt.ylabel('Density')
    plt.show()


# Plot the distribution of loss severity per Loss
def plot_lossSeverityPerLoss(lossSeries):
    # Define boundaries of bins
    bins = np.logspace(np.log10(min(lossSeries)), np.log10(max(lossSeries)), num=30)
    # Create the histogram with logarithmically spaced bins
    plt.hist(lossSeries, bins=bins, edgecolor='black', density=True)
    plt.title('Distribution of Loss severity per Loss')
    plt.xlabel('Loss Severity')
    plt.ylabel('Density')
    plt.show()


# Plot the quantile to quantile plot to assess goodness of fit
def qq_plot(data, params, distribution, distribution_name):
    # Sort the data to get the empirical losses
    sorted_losses = np.sort(data)

    # Calculate the corresponding probabilities for the theoretical quantiles
    m = len(sorted_losses)
    probabilities = (np.arange(m) + 0.5) / m

    # Calculate the corresponding theoretical quantiles
    theoretical_quantiles = distribution.ppf(probabilities, *params)

    # Generate the Q-Q plot
    plt.figure()
    plt.plot(sorted_losses, theoretical_quantiles, 'o', label='Data Quantiles')
    min_val = max(0, min(np.min(sorted_losses), np.min(theoretical_quantiles)))
    max_val = max(np.max(sorted_losses), np.max(theoretical_quantiles))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Fit')  # Diagonal line
    plt.xlabel('Empirical Quantiles')
    plt.ylabel('Theoretical Quantiles')
    plt.title(f'{distribution_name} Distribution Q-Q Plot')
    plt.legend()
    plt.show()


# Histogram of simulation results
def plot_cumulative_losses(sim_results, title):

    plt.hist(sim_results, bins=50, density=True)

    plt.title(title)
    plt.xlabel('Yearly Loss')
    plt.ylabel('Relative Frequency')

    plt.show()


# def exAnte_boxPlots(data):
#    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # Boxplot for 'Yearly Frequency'
#    sns.boxplot(y=data['Yearly Frequency'], ax=ax[0])
#    ax[0].set_title('Boxplot for Yearly Frequency')

   # Boxplot for 'Typical Loss'
#    sns.boxplot(y=data['Typical Loss'], ax=ax[1])
#    ax[1].set_title('Boxplot for Typical Loss')

    # Boxplot for 'Extreme Loss'
#    sns.boxplot(y=data['Extreme Loss'], ax=ax[2])
#    ax[2].set_title('Boxplot for Extreme Loss')

#    plt.tight_layout()
#    plt.show()
