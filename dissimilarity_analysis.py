from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import matplotlib.pyplot as plt

with open('spectra_mat.npy', 'rb') as f:
    spectra = np.load(f)
with open('wavelengths.npy', 'rb') as f:
    wavelength_vect = np.load(f)

# generate dissimilarity matrix for L1, "cityblock" or "taxicab"
d1 = pairwise_distances(spectra[:40,], metric='manhattan')

plt.figure()
plt.imshow(d1)
plt.title('City Block (L1)');
plt.colorbar()
plt.show()

plt.figure()
for i in range(40):
    plt.plot(wavelength_vect, spectra[i,])
plt.show()

np.min(np.max(X1, axis = 1))
