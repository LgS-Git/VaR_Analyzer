import numpy as np
import pandas as pd
from scipy import stats
from src.distribution_estimation import distribution_mapping_discrete, distribution_mapping_continuous, fit_discrete_distributions
from src.visualizations import plot_cumulative_losses


def perform_exAnte_simulations(df, n):
    # Preallocate a NumPy array for all simulation results
    all_results = np.zeros((n, len(df) + 1))

    for index, row in df.iterrows():
        scenario = row['Scenario']
        L_typ = row['Typical Loss']
        L_ext = row['Extreme Loss']
        poisson_param = row['Yearly Frequency']

        # Calculate log-normal parameters
        sigma = 0.5*(-stats.norm.ppf(0.99) + np.sqrt(stats.norm.ppf(0.99) ** 2 - 4*np.log(L_typ/L_ext)))
        mu = np.log(L_typ) + sigma ** 2

        # Simulate yearly occurence from Poisson distribution and loss from log-normal distribution
        occurrence = np.random.poisson(poisson_param, n)
        loss = np.random.lognormal(mean=mu, sigma=sigma, size=n)

        # Calculate expected loss for each simulation
        expected_loss = occurrence * loss

        # Store results
        all_results[:, index] = expected_loss

    # Calculate cumulative loss for each simulation step
    all_results[:, -1] = np.sum(all_results, axis=1)

    # Create DataFrame from results
    results_df = pd.DataFrame(all_results, columns=list(df["Scenario"]) + ["Cumulative Loss"])

    # Save to Excel
    results_df.to_excel("data/exAnte_simulation_results.xlsx", index=False)

    # Plot histogram of cumulative losses
    plot_cumulative_losses(results_df['Cumulative Loss'], 'Ex-Ante Yearly Loss')

    # Calculate VaR
    VaR_99_9_exAnte = round(np.percentile(all_results[:, -1], 0.1), 2)
    print(f'Ex-Ante VaR (99.9%): {VaR_99_9_exAnte}')


def perform_exPost_simulations(data, discrete_dist_name, continuous_dist_name, n):
    # Get the distribution objects from the mapping dictionaries
    discrete_dist = distribution_mapping_discrete[discrete_dist_name]
    continuous_dist = distribution_mapping_continuous[continuous_dist_name]

    # Estimate the parameters
    discrete_dist_name, params_discrete = fit_discrete_distributions(data['Time Stamp'].dt.year.value_counts(), discrete_dist_name)
    params_continuous = continuous_dist.fit(data['Loss'])

    # Draw all values from the discrete distribution at once
    num_values_all = discrete_dist.rvs(*params_discrete, size=n)

    # Find the maximum number of losses in a single simulation step
    max_losses = np.max(num_values_all)

    # Initialize an array to store all losses
    all_losses = np.zeros((n, max_losses))

    # Draw all (and more) values from the continuous distribution at once
    max_num_values = np.max(num_values_all)
    all_loss_values = continuous_dist.rvs(*params_continuous, size=(n, max_num_values))

    # Use only the number of values indicated by the discrete distribution
    for i in range(n):
        num_values = num_values_all[i]
        all_losses[i, :num_values] = all_loss_values[i, :num_values]

    # Above, unused random numbers are drawn, but it still seems to be more efficient than drawing random numbers inside the for loop
    # For memory saving however, activate the for loop below (Instead of: for loop above + all at once drawing from continuous dist)
    # for i in range(n):
        # Draw that many values from the continuous distribution
    #    num_values = num_values_all[i]
    #    loss_values = continuous_dist.rvs(*params_continuous, size=num_values)

        # Store the losses in the array
    #    all_losses[i, :num_values] = loss_values

    # Compute the cumulative losses
    cumulative_losses = np.sum(all_losses, axis=1)

    # Create the DataFrame
    results_df = pd.DataFrame(all_losses, columns=[f'Loss {i+1}' for i in range(max_losses)])
    results_df['Cumulative Loss'] = cumulative_losses

    # Save to Excel
    results_df.to_excel("data/exPost_simulation_results.xlsx", index=False)

    # Plot histogram of cumulative losses
    plot_cumulative_losses(results_df['Cumulative Loss'], 'Ex-Post Yearly Loss')

    # Calculate VaR
    VaR_99_9_exPost = round(np.percentile(cumulative_losses, 0.1), 2)
    print(f'Ex-Post VaR (99.9%): {VaR_99_9_exPost}')


def combine_simulation_results(exAnte_simulation, exPost_simulation, weight_exante):
    # Read the simulation results from the Excel files
    exante_df = pd.read_excel(exAnte_simulation)
    expost_df = pd.read_excel(exPost_simulation)

    # Determine the lengths of both simulations
    len_exante = len(exante_df['Cumulative Loss'])
    len_expost = len(expost_df['Cumulative Loss'])

    # Calculate the proportion of data to use from each simulation based on the weight and available data
    n_exante = min(int(len_expost * weight_exante / (1 - weight_exante)), len_exante)
    n_expost = min(int(len_exante * (1 - weight_exante) / weight_exante), len_expost)

    # Draw the samples from each simulation
    samples_exante = exante_df['Cumulative Loss'].sample(n_exante)
    samples_expost = expost_df['Cumulative Loss'].sample(n_expost)

    # Combine the samples
    combined_samples = pd.concat([samples_exante, samples_expost])

    print(f"Used {len(samples_exante)} samples from ex-ante simulation and {len(samples_expost)} samples from ex-post simulation:")

    # Plot combined distribution
    plot_cumulative_losses(combined_samples, 'Combined Yearly Loss')

    # Calculate VaR
    VaR_99_9_combined = round(np.percentile(combined_samples, 0.1), 2)
    print(VaR_99_9_combined)

    return VaR_99_9_combined
