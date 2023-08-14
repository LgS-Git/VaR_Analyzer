import ipywidgets as widgets
import src.distribution_estimation


def create_discrete_dropdown():

    dropdown_discrete = widgets.Dropdown(
        options=['Poisson', 'Geometric', 'Negative Binomial'],
        value='Poisson',
        description='Discrete Distribution:',
        layout={'width': 'max-content'}
    )
    return dropdown_discrete


def create_continuous_dropdown():

    dropdown_continuous = widgets.Dropdown(
        options=['Lognormal', 'Normal', 'Exponential', 'Gamma', 'Loggamma', 'Beta', 'Logistic', 'Gumbel_r', 'Gumbel_l', 'Weibull', 'Pareto', 'Cauchy'],
        value='Lognormal',
        description='Continuous Distribution:',
        layout={'width': 'max-content'}
    )
    return dropdown_continuous


def on_dropdown_discrete_change(change, data):
    if change['type'] == 'change' and change['name'] == 'value':
        params = src.distribution_estimation.discrete_fit_tests(data, change['new'])


def on_dropdown_continuous_change(change, data):
    if change['type'] == 'change' and change['name'] == 'value':
        params = src.distribution_estimation.continuous_fit_tests(data, change['new'])
