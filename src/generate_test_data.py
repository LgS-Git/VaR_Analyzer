if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    # Seed
    # np.random.seed(0)

    # Schadensfalldatenbank

    n = 1000

    # Generate list of events
    event_types = ['Fraud Case', 'IT System Crash', 'Employee Misconduct', 'Cybersecurity Breach', 'Legal Violation']
    event_names = list(np.random.choice(event_types, size=n))

    # Generate a list of dates
    all_dates = pd.date_range(start='2000-01-01', end='2023-12-31').strftime('%Y-%m-%d')

    # Randomly select n dates from all_dates and sort them
    time_stamps = sorted(np.random.choice(all_dates, size=n, replace=False))

    # log-normal distribution
    print('// Ex Post //')
    damages = np.random.lognormal(mean=10, sigma=0.6, size=n)
    print('Min: ' + str(min(damages)))
    print('Median: ' + str(np.median(damages)))
    print('Max: ' + str(max(damages)))
    damages = np.round(damages, 2)

    df_DB = pd.DataFrame({
        'Event Name': event_names,
        'Time Stamp': time_stamps,
        'Loss': damages
    })

    df_DB.to_excel("data/exPost_example_data.xlsx", index=False)

    # Szenario Analyse (Workshopergebnisse)

    # List of scenarios
    scenario_types = ['Fraud Case', 'IT System Crash', 'Employee Misconduct', 'Cybersecurity Breach', 'Legal Violation',
                      'Physical Damage', 'Weather Event', 'Cybersecurity Lack', 'Lawsuit', 'Large Operational Loss']

    n = len(scenario_types)

    # Frequency of events per year
    frequencies = np.round(np.random.uniform(low=1, high=3, size=n), 2)

    # Typical and extreme damages
    print('// Ex Ante //')
    typical_damages = np.round(np.random.uniform(low=15000, high=50000, size=n), 2)
    extreme_damages = np.round(np.random.uniform(low=100000, high=300000, size=n), 2)
    print(f'Typical Loss Range: [{min(typical_damages)}, {max(typical_damages)}]')
    print(f'Extreme Loss Range: [{min(extreme_damages)}, {max(extreme_damages)}]')

    df_scenarios = pd.DataFrame({
        'Scenario': scenario_types,
        'Yearly Frequency': frequencies,
        'Typical Loss': typical_damages,
        'Extreme Loss': extreme_damages
    })

    df_scenarios.to_excel("data/exAnte_example_data.xlsx", index=False)
