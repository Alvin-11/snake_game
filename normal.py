import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Data
data = [
    0.88, 0.06, 5.63, 0.75, 0.21, 0.83, 1.20, 0.38, 1.33, 1.20, 0.71, 1.00,
    0.15, 0.40, 0.19, 1.11, 4.00, 0.75, 0.25, 0.17, 2.67, 1.43, 1.33, 1.30,
    1.67, 1.00, 1.00, 2.00, 1.43, 0.87, 0.40, 0.53, 1.11, 0.40, 0.00, 1.67,
    2.22, 5.00, 1.50, 0.20, 3.33, 2.00, 0.67, 0.33, 0.24, 0.93, 0.83, 0.50,
    0.35, 0.31
]

# Plotting the histogram
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(data, bins=10, edgecolor='black')
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Plotting the Q-Q plot
plt.subplot(1, 2, 2)
stats.probplot(data, dist="norm", plot=plt)
plt.title('Q-Q Plot')

plt.tight_layout()
plt.show()

# Shapiro-Wilk Test
shapiro_test = stats.shapiro(data)
print(f'Shapiro-Wilk Test statistic: {shapiro_test.statistic:.4f}, p-value: {shapiro_test.pvalue:.4f}')