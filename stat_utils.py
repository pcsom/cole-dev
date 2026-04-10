import numpy as np
import scipy.stats as stats

def corrected_paired_ttest_counted(d_mean, d_var, n, n_train, n_test):
    # Corrected variance
    corrected_var = (1.0 / n + n_test / n_train) * d_var
    
    # t-statistic
    if corrected_var <= 0:
        return 0.0, 1.0
    
    t_stat = d_mean / np.sqrt(corrected_var)
    
    # Degrees of freedom
    df = n - 1
    
    # One-tailed p-value: tests H1: model2 > model1 (i.e., difference < 0)
    # If t_stat < 0, model2 is better (negative difference means model1 < model2)
    if t_stat < 0:
        # Model2 is better, report lower tail probability
        p_value = stats.t.cdf(t_stat, df)
    else:
        # Model1 is better or equal, report upper tail (1 - prob)
        p_value = 1 - stats.t.cdf(t_stat, df)
    
    return t_stat, p_value


def corrected_paired_ttest(differences, n_train, n_test):
    """
    One-tailed corrected paired t-test accounting for non-independence in CV.
    Tests if model2 is significantly better than model1.
    
    From Nadeau & Bengio (2003):
    t = mean(d) / sqrt( (1/n + n_test/n_train) * var(d) )
    
    Args:
        differences: Array of performance differences (Model1_R² - Model2_R²)
        n_train: Training set size used
        n_test: Test set size
    
    Returns:
        t-statistic, p-value (one-tailed: tests if model2 > model1)
    """
    d_mean = np.mean(differences)
    d_var = np.var(differences, ddof=1)
    n = len(differences)
    
    # Corrected variance
    corrected_var = (1.0 / n + n_test / n_train) * d_var
    
    # t-statistic
    if corrected_var <= 0:
        return 0.0, 1.0
    
    t_stat = d_mean / np.sqrt(corrected_var)
    
    # Degrees of freedom
    df = n - 1
    
    # One-tailed p-value: tests H1: model2 > model1 (i.e., difference < 0)
    # If t_stat < 0, model2 is better (negative difference means model1 < model2)
    if t_stat < 0:
        # Model2 is better, report lower tail probability
        p_value = stats.t.cdf(t_stat, df)
    else:
        # Model1 is better or equal, report upper tail (1 - prob)
        p_value = 1 - stats.t.cdf(t_stat, df)
    
    return t_stat, p_value



# main method
if __name__ == "__main__":
    # Example usage
    d_mean = 0.7092-0.7033
    d_var = (0.0114) ** 2
    n = 879
    n_train = int(15625*0.9)
    n_test = int(15625*0.1)
    
    t_stat, p_value = corrected_paired_ttest_counted(d_mean, d_var, n, n_train, n_test)
    print(f"T-statistic: {t_stat}, P-value: {p_value}")