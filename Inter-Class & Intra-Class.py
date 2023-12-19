import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
# Generate sample data for two classes
np.random.seed(0)
class1 = np.random.normal(4, 1.5, 100)
class2 = np.random.normal(8, 1.5, 100)

# Calculate mean of each class
mean1 = np.mean(class1)
mean2 = np.mean(class2)

# Calculate overall mean
overall_mean = np.mean(np.concatenate((class1, class2)))

# Plotting
plt.figure(figsize=(10, 6))

# Plot class 1
plt.hist(class1, bins=15, alpha=0.7, label="Class 1")
plt.axvline(mean1, color='blue', linestyle='dashed', linewidth=2)

# Plot class 2
plt.hist(class2, bins=15, alpha=0.7, label="Class 2")
plt.axvline(mean2, color='orange', linestyle='dashed', linewidth=2)

# Plot overall mean
plt.axvline(overall_mean, color='red', linestyle='dashed', linewidth=2, label='Overall Mean')

# Annotations
plt.text(mean1, 5, 'Mean of Class 1', color='blue', horizontalalignment='center')
plt.text(mean2, 5, 'Mean of Class 2', color='orange', horizontalalignment='center')
plt.text(overall_mean, 10, 'Overall Mean', color='red', horizontalalignment='center')

# Labels and title
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Class Separation: Inter-Class and Intra-Class Variance')
plt.legend()

# Show plot
plt.show()
