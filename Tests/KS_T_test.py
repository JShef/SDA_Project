# sources used:
# https://www.geeksforgeeks.org/how-to-conduct-a-two-sample-t-test-in-python/
# https://www.tutorialspoint.com/how-to-conduct-a-two-sample-t-test-in-python
# https://www.datacamp.com/tutorial/an-introduction-to-python-t-tests

# https://www.statology.org/kolmogorov-smirnov-test-python/
# https://www.statology.org/kolmogorov-smirnov-test-python/
# https://www.listendata.com/2019/07/KS-Statistics-Python.html

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm



def KS_and_T_test(sample1, sample2):
    # Kolmgorov-Smirnov Test (Non-parametric)
    def ks_test(sample1, sample2):
        all_points = sorted(sample1 + sample2)
        cdf1_interp = [sum(x <= point for x in sorted(sample1)) / len(sorted(sample1)) for point in all_points]
        cdf2_interp = [sum(x <= point for x in sorted(sample2)) / len(sorted(sample2)) for point in all_points]
        ks_stat = max(abs(c1 - c2) for c1, c2 in zip(cdf1_interp, cdf2_interp))
        p_value = 2 * np.exp(-2 * (ks_stat * np.sqrt(len(sample1) * len(sample2) / (len(sample1) + len(sample2)))) ** 2)

        return p_value

    # Two-sample t-test (Parametric)
    def t_test(sample1, sample2):
        pooled_std = np.sqrt(((len(sample1) - 1) * np.var(sample1) + (len(sample2) - 1) * np.var(sample2)) / (len(sample1) + len(sample2) - 2))
        t_stat = (np.mean(sample1) - np.mean(sample2)) / (pooled_std * np.sqrt(1 / len(sample1) + 1 / len(sample2)))
        p_value = 2 * (1 - norm.cdf(np.abs(t_stat), loc=0, scale=1))

        return p_value

    sample_sizes = np.arange(2, 102, 2)
    repetitions = 10 ** 3

    ks_accuracies = list()
    ttest_accuracies = list()

    for n in sample_sizes:
        ks_correct = 0
        ttest_correct = 0
        for _ in range(repetitions):
            # same distributions
            sample1 = np.random.normal(0, 1, n)
            sample2 = np.random.normal(0, 1, n)
            if ks_test(sample1, sample2) > 0.05:  # doesnt reject H0
                ks_correct += 1
            if t_test(sample1, sample2) > 0.05:  # doenst reject H0
                ttest_correct += 1

            #  different distributions
            sample1 = np.random.normal(0, 1, n)
            sample2 = np.random.normal(1, 1, n)
            if ks_test(sample1, sample2) < 0.05:  # reject H0
                ks_correct += 1
            if t_test(sample1, sample2) < 0.05:  # reject H0
                ttest_correct += 1

        ks_accuracies.append(0.5 * ks_correct / repetitions)
        ttest_accuracies.append(0.5 * ttest_correct / repetitions)

    plt.figure(figsize=(15, 8))
    plt.plot(sample_sizes, ks_accuracies, label="KS test accuracy")
    plt.plot(sample_sizes, ttest_accuracies, label="T-test accuracy")
    plt.xlabel("Sample size")
    plt.ylabel("Accuracy")
    plt.title("Accuracy of two-sample tests")
    plt.legend()
    plt.show()

KS_and_T_test(sample1, sample2)
