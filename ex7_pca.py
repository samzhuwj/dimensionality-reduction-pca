import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from mpl_toolkits.mplot3d import Axes3D
from skimage import io
from skimage import img_as_float

from featureNormalize import feature_normalize
from pca import pca 
from runkMeans import run_kmeans
from projectData import project_data
from recoverData import recover_data
from displayData import display_data
from kMeansInitCentroids import kmeans_init_centroids
from runkMeans import run_kmeans, draw_line


plt.ion()
np.set_printoptions(formatter={'float': '{: 0.6f}'.format})


# ===================== Part 1: Load Example Dataset =====================
# We start this exercise by using a small dataset that is easily to
# visualize
print('Visualizing example dataset for PCA.')

# The following command loads the dataset.
data = scio.loadmat('ex7data1.mat')
X = data['X']

# Visualize the example dataset
plt.figure()
plt.scatter(X[:, 0], X[:, 1], facecolors='none', edgecolors='b', s=20)
plt.axis('equal')
plt.axis([0.5, 6.5, 2, 8])

input('Program paused. Press ENTER to continue')


# ===================== Part 2: Principal Component Analysis =====================
# You should now implement PCA, a dimension reduction technique. You
# should complete the code in pca.py
print('Running PCA on example dataset.')

# Before running PCA, it is important to first normalize X
X_norm, mu, sigma = feature_normalize(X)

# Run PCA
U, S = pca(X_norm)
draw_line(mu, mu+1.5*S[0]*U[:, 0])
draw_line(mu, mu+1.5*S[1]*U[:, 1])

print('Top eigenvector: \nU[:, 0] = {}'.format(U[:, 0]))
print('You should expect to see [-0.707107 -0.707107]')

input('Program paused. Press ENTER to continue')


# ===================== Part 3: Dimension Reduction =====================
# You should now implement the projection step to map the data onto the
# first k eigenvectors. The code will then plot the data in this reduced
# dimensional space. This will show you what the data looks like when
# using only the correspoding eigenvectors to reconstruct it.

# You should complete the code in projectData.py
print('Dimension reductino on example dataset.')

# Plot the normalized dataset (returned from pca)
plt.figure()
plt.scatter(X_norm[:, 0], X_norm[:, 1], facecolors='none', edgecolors='b', s=20)
plt.axis('equal')
plt.axis([-4, 3, -4, 3])

# Project the data onto K = 1 dimension
K = 1
Z = project_data(X_norm, U, K)
print('Projection of the first example: {}'.format(Z[0]))
print('(this value should be about 1.481274)')

X_rec = recover_data(Z, U, K)

print('Approximation of the first example: {}'.format(X_rec[0]))
print('(this value should be about [-1.047419 -1.047419])')

# Draw lines connecting the projected points to the original points
plt.scatter(X_rec[:, 0], X_rec[:, 1], facecolors='none', edgecolors='r', s=20)
for i in range(X_norm.shape[0]):
    draw_line(X_norm[i], X_rec[i])

input('Program paused. Press ENTER to continue')


# ===================== Part 4: Loading and Visualizing Face Data =====================
# We start the exercise by first loading and visualizing the dataset.
# The following code will load the dataset into your environment
print('Loading face dataset.')

# Load Face dataset
data = scio.loadmat('ex7faces.mat')
X = data['X']
display_data(X[0:100])

input('Program paused. Press ENTER to continue')


# ===================== Part 5: PCA on Face Data: Eigenfaces =====================
# Run PCA and visualize the eigenvectors which are in this case eigenfaces
# We display the first 36 eigenfaces.
print('Running PCA on face dataset.\n(this might take a minute or two ...)')

# Before running PCA, it is important to first normalize X by subtracting
# the mean value from each feature
X_norm, mu, sigma = feature_normalize(X)

# Run PCA
U, S = pca(X_norm)

# Visualize the top 36 eigenvectors found
display_data(U[:, 0:36].T)

input('Program paused. Press ENTER to continue')


# ===================== Part 6: Dimension Reduction for Faces =====================
# Project images to the eigen space using the top k eigenvectors
# If you are applying a machine learning algorithm
print('Dimension reduction for face dataset.')

K = 100
Z = project_data(X_norm, U, K)

print('The projected data Z has a shape of: {}'.format(Z.shape))

input('Program paused. Press ENTER to continue')


# =========== Part 7: Visualization of Faces after PCA Dimension Reduction ===========
# Project images to the eigen space using the top K eigen vectors and
# visualize only using those K dimensions
# Compare to the original input, which is also displayed
print('Visualizing the projected (reduced dimension) faces.')

K = 100
X_rec = recover_data(Z, U, K)

# Display normalized data
display_data(X_norm[0:100])
plt.title('Original faces')
plt.axis('equal')

# Display reconstructed data from only k eigenfaces
display_data(X_rec[0:100])
plt.title('Recovered faces')
plt.axis('equal')

input('Program paused. Press ENTER to continue')


# ===================== Part 8(a): PCA for Visualization =====================
# One useful application of PCA is to use it to visualize high-dimensional
# data. In the last K-Means exercise you ran K-Means on 3-dimensional
# pixel colors of an image. We first visualize this output in 3D, and then
# apply PCA to obtain a visualization in 2D.

# Reload the image from the previous exercise and run K-Means on it
# For this to work, you need to complete the K-Means assignment first
image = io.imread('bird_small.png')
image = img_as_float(image)

img_shape = image.shape

X = image.reshape((img_shape[0]*img_shape[1], 3))
K = 16
max_iters = 10
initial_centroids = kmeans_init_centroids(X, K)
centroids, idx = run_kmeans(X, initial_centroids, max_iters, False)

# Sample 1000 random indices (since working with all the data is
# too expensive. If you have a fast computer, you may increase this.
selected = np.random.randint(X.shape[0], size=1000)

# Visualize the data and centroid memberships in 3D
cm = plt.cm.get_cmap('RdYlBu')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[selected, 0], X[selected, 1], X[selected, 2], c=idx[selected].astype(np.float64), s=15, cmap=cm, vmin=0, vmax=K)
plt.title('Pixel dataset plotted in 3D. Color shows centroid memberships')

input('Program paused. Press ENTER to continue')


# ===================== Part 8(b): PCA for Visualization =====================
# Use PCA to project this cloud to 2D for visualization
X_norm, mu, sigma = feature_normalize(X)

# PCA and project the data to 2D
U, S = pca(X_norm)
Z = project_data(X_norm, U, 2)

# Plot in 2D
plt.figure()
plt.scatter(Z[selected, 0], Z[selected, 1], c=idx[selected].astype(np.float64), s=15, cmap=cm)
plt.title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction')

input('ex7_pca Finished. Press ENTER to exit')
