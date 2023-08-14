from src.visualizations import plot_lossesPerYear, plot_lossSeverityPerLoss


def exploratory_statistics_discrete(data):
    mean = data.mean()
    median = data.median()
    std_dev = data.std()
    skewness = data.skew()
    kurtosis = data.kurtosis()

    print("Mean:", mean)
    print("Median:", median)
    print("Standard Deviation:", std_dev)
    print("Skewness:", skewness)
    print("Kurtosis:", kurtosis)

    plot_lossesPerYear(data)


def exploratory_statistics_continuous(data):
    mean = data.mean()
    median = data.median()
    std_dev = data.std()
    skewness = data.skew()
    kurtosis = data.kurtosis()

    print("Mean:", mean)
    print("Median:", median)
    print("Standard Deviation:", std_dev)
    print("Skewness:", skewness)
    print("Kurtosis:", kurtosis)

    plot_lossSeverityPerLoss(data)
