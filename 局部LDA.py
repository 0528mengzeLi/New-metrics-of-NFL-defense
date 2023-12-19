import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
# Creating synthetic data to simulate the Yale Faces Dataset
np.random.seed(0)

# Generating data for three facial features: eyes, nose, mouth
# Each feature has two classes: neutral and expressive
eyes_neutral = np.random.normal(0, 1, (20, 2)) + np.array([2, 8])
eyes_expressive = np.random.normal(0, 1, (20, 2)) + np.array([5, 8])

nose_neutral = np.random.normal(0, 1, (20, 2)) + np.array([2, 5])
nose_expressive = np.random.normal(0, 1, (20, 2)) + np.array([5, 5])

mouth_neutral = np.random.normal(0, 1, (20, 2)) + np.array([2, 2])
mouth_expressive = np.random.normal(0, 1, (20, 2)) + np.array([5, 2])

# Plotting
plt.figure(figsize=(12, 4))

# Eyes region
plt.subplot(1, 3, 1)
plt.scatter(eyes_neutral[:, 0], eyes_neutral[:, 1], color='blue', label='Neutral')
plt.scatter(eyes_expressive[:, 0], eyes_expressive[:, 1], color='green', label='Expressive')
plt.title("Eyes Region")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

# Nose region
plt.subplot(1, 3, 2)
plt.scatter(nose_neutral[:, 0], nose_neutral[:, 1], color='blue', label='Neutral')
plt.scatter(nose_expressive[:, 0], nose_expressive[:, 1], color='green', label='Expressive')
plt.title("Nose Region")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

# Mouth region
plt.subplot(1, 3, 3)
plt.scatter(mouth_neutral[:, 0], mouth_neutral[:, 1], color='blue', label='Neutral')
plt.scatter(mouth_expressive[:, 0], mouth_expressive[:, 1], color='green', label='Expressive')
plt.title("Mouth Region")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

plt.suptitle("Conceptual Visualization of Yale Faces Dataset")
plt.show()
