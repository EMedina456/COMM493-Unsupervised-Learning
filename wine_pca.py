import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # This is needed for 3d plotting

# Load the data
raw_data = pd.read_csv('wine.csv')
# Extract labels from the cultivar column
labels = raw_data['Cultivar']
# Drop the cultivar column from the data
data = raw_data.drop('Cultivar', axis=1)
# Normalize the data along the columns
data = (data - data.mean()) / data.std()
# Run PCA on the data and generate skree plot
pca = PCA()
pca.fit(data)
# Save the coefficients to a DataFrame
coefficients = pd.DataFrame(pca.components_, columns=data.columns)
print(data.shape)
print(pca.explained_variance_ratio_)

# Plot the skree plot
plt.plot(np.arange(1, 14), pca.explained_variance_ratio_)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
# Make the x-axis increment by 1
plt.xticks(np.arange(1, 14))
plt.ylabel('Explained Variance Ratio')

# Reduce data to 3 dimensions by multiplying by the first 3 coefficients
data_3d = np.dot(data, pca.components_[:3].T)

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Color the points by their labels
colors = ['r', 'g', 'b']
for i in range(3):
    # Select the data points that correspond to the label 'i+1'
    label_mask = labels == i+1
    ax.scatter(data_3d[label_mask, 0], data_3d[label_mask, 1], data_3d[label_mask, 2], c=colors[i], label=f'Cultivar {i+1}')

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.legend()
plt.title('3D PCA Scatter Plot')

# Display the plot
plt.show()


''' NOTE: How to use this script
First of all, this data has not left any of the entries out for testing. 
In reality we should take out an entry from each class before doing the PCA.

The 'coefficients' DataFrame contains the coefficients of the principal components.
This is what you would apply to unknown/test data to reduce its dimensionality.
'data_3d' contains the data reduced to 3 dimensions FOR THIS SPECIFIC EXAMPLE.
'data_3d' is the matrix multiplication product of the original data and the first 3 coefficients vectors (as a matrix).
'''