import matplotlib.pyplot as plt
import numpy as np

# Random data
np.random.seed(42)
data = np.random.randn(100)
data2 = np.random.rand(10)
categories = ["A","B","C","D","E"]

fig, axs = plt.subplots(5, 2, figsize=(12, 18))
axs = axs.ravel()

# 1. Line Plot
axs[0].plot(np.arange(50), np.cumsum(np.random.randn(50)))
axs[0].set_title("Line Plot")

# 2. Scatter Plot
axs[1].scatter(np.random.randn(100), np.random.randn(100))
axs[1].set_title("Scatter Plot")

# 3. Bar Chart
axs[2].bar(categories, np.random.randint(1, 10, size=5))
axs[2].set_title("Bar Chart")

# 4. Histogram
axs[3].hist(data, bins=15)
axs[3].set_title("Histogram")

# 5. Pie Chart
axs[4].pie(np.abs(np.random.randn(5)), labels=categories, autopct='%1.1f%%')
axs[4].set_title("Pie Chart")

# 6. Boxplot
axs[5].boxplot([data, data2])
axs[5].set_title("Boxplot")

# 7. Area Plot
axs[6].plot(np.arange(10), data2)
axs[6].fill_between(np.arange(10), data2, alpha=0.3)
axs[6].set_title("Area Plot")

# 8. Heatmap
heat = np.random.rand(5, 5)
im = axs[7].imshow(heat, aspect='auto')
axs[7].set_title("Heatmap")
fig.colorbar(im, ax=axs[7])

# 9. Violin Plot
axs[8].violinplot([data, data2], showmeans=True)
axs[8].set_title("Violin Plot")

# 10. Stem Plot
axs[9].stem(np.arange(10), np.random.rand(10), use_line_collection=True)
axs[9].set_title("Stem Plot")

plt.tight_layout()
plt.show()
