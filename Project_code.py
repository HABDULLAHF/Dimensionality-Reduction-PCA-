import os
from glob import glob

import cv2

from skimage import io, color
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from matplotlib.image import imread
import numpy as np
import xarray as xr
import rioxarray as rxr
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep

landsat_path = os.getcwd()
paths = glob(os.path.join(landsat_path, "Bands", "*BAND*.TIF"))
paths.sort()

def open_clean_bands(band_path):     
    return rxr.open_rasterio(band_path, masked=True).squeeze()

all_bands = []
for i, aband in enumerate(paths):
    all_bands.append(open_clean_bands(aband))
    # Assign a band number to the new xarray object
    all_bands[i]["band"]=i+1


#concat
landsat__xr = xr.concat(all_bands,  dim="band") 

#print(landsat__xr)



landsat__xr.plot.imshow(col="band", col_wrap=3,cmap="Greys_r")
plt.show()



ep.plot_rgb(landsat__xr.values,
            rgb=[3, 2, 1],
            title="Landsat RGB Image\n Linear Stretch Applied",
            stretch=True,
            str_clip=1)
plt.show()

img = cv2.cvtColor(cv2.imread(paths[6]) , cv2.COLOR_BGR2RGB)

print(img.shape)

dim = (400, 400)
  
# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
print(resized.shape)

plt.imshow(resized)

lina_gray = color.rgb2gray(resized)

plt.imshow(lina_gray)

r , g, b = cv2.split(resized)

r,g,b = r/255 , g/255 , b/255
r = np.array(r)
g = np.array(g)
b = np.array(b)


class convers_pca():
    def __init__(self, no_of_components):
        self.no_of_components = no_of_components
        self.eigen_values = None
        self.eigen_vectors = None
        
    def transform(self, x):
        return np.dot(x - self.mean, self.projection_matrix.T)
    
    def inverse_transform(self, x):
        return np.dot(x, self.projection_matrix) + self.mean
    
    def fit(self, x):
        self.no_of_components = x.shape[1] if self.no_of_components is None else self.no_of_components
        self.mean = np.mean(x, axis=0)
        
        cov_matrix = np.cov(x - self.mean, rowvar=False)
        
        self.eigen_values, self.eigen_vectors = np.linalg.eig(cov_matrix)
        self.eigen_vectors = self.eigen_vectors.T
        
        self.sorted_components = np.argsort(self.eigen_values)[::-1]
        
        self.projection_matrix = self.eigen_vectors[self.sorted_components[:self.no_of_components]]
        self.explained_variance = self.eigen_values[self.sorted_components]
        self.explained_variance_ratio = self.explained_variance / self.eigen_values.sum()
        
        
pca_components = 100

pca_r = convers_pca(no_of_components = pca_components)
pca_r.fit(r)
reduced_r = pca_r.transform(r)

pca_g = convers_pca(no_of_components = pca_components)
pca_g.fit(g)
reduced_g = pca_r.transform(g)

pca_b = convers_pca(no_of_components = pca_components)
pca_b.fit(b)
reduced_b = pca_r.transform(b)

combined = np.array([reduced_r,reduced_g,reduced_b])

print(combined.shape)

reconstructed_r = pca_r.inverse_transform(reduced_r)
reconstructed_g = pca_g.inverse_transform(reduced_g)
reconstructed_b = pca_b.inverse_transform(reduced_b)

print(reconstructed_b.shape)

img_reconstructed = (cv2.merge((reconstructed_r,reconstructed_g,reconstructed_b)))

plt.imshow(img_reconstructed)

reconstructed_gray = color.rgb2gray(img_reconstructed)

def mse(actual, predicted):
    return np.square(np.subtract(np.array(actual), np.array(predicted))).mean()

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

print('Mean Sqaure Error = '+ str(mse(reconstructed_gray,lina_gray)))
print('Root MSE = '+ str(rmse(reconstructed_gray,lina_gray)))