# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Create dataset
#Builds a 12×6 table: 12 countries each with 6 features: GDP, Literacy Rate, Life Expectancy, Internet Access, CO₂ Emissions, and Education Index.
data = {
    "GDP": [55,30,62,15,45,20,58,25,48,12,40,35],
    "Literacy": [98,85,99,70,92,75,97,80,94,65,90,88],
    "LifeExp": [82,74,84,65,79,68,83,70,80,60,77,75],
    "Internet": [92,65,95,40,85,50,90,60,88,35,78,72],
    "CO2": [8.1,4.0,9.5,2.2,6.7,3.0,8.9,3.8,7.2,1.5,5.8,5.0],
    "Education": [9.0,6.5,9.3,4.8,8.1,5.5,9.1,6.0,8.5,4.2,7.5,7.0]
}

df = pd.DataFrame(data)

# Step 2: Center the data
#Centering shifts each feature so its mean = 0. This is required before PCA so no feature dominates just because of its scale of measurement.
X = df.values  # converts DataFrame → raw NumPy array
print("the first matrix of data (X):\n", X)
mean = np.mean(X, axis=0) # computes the mean of each column
Xc = X - mean # subtracts mean from every value
print("Centered Data (Xc):\n", Xc) #prints the centered data

# Step 3: Covariance matrix
#Builds a 6×6 matrix that measures how much each pair of features varies together. High covariance between GDP and Internet Access, for example, means wealthy countries tend to have better internet.
C = np.cov(Xc, rowvar=False)
print("\nCovariance Matrix:\n", C) #printing that matrix
##The covariance matrix shows strong positive relationships between GDP, literacy, and education.”

# Step 4: Eigenvalues and eigenvectors
#Decomposes the covariance matrix to find:
#Eigenvectors → the new axes (principal component directions)
#Eigenvalues → how much variance each axis takes
eigenvalues, eigenvectors = np.linalg.eig(C)
print("\nEigenvalues:\n", eigenvalues)  #printing the Eigenvalues
print("\nEigenvectors:\n", eigenvectors) #printing the Eigenvectors


# Step 5: Sort eigenvalues
#Ranks the components so PC1 explains the most variance, PC2 the second most, etc.
idx = np.argsort(eigenvalues)[::-1] # indices sorted from largest to smallest
eigenvalues = eigenvalues[idx] # reorder eigenvalues
eigenvectors = eigenvectors[:, idx] # reorder eigenvectors to match

# Step 6: Select first 2 principal components
#Takes only the first 2 eigenvectors (columns), forming a 6×2 projection matrix. This is the "dimensionality reduction" step — from 6D down to 2D.
PC = eigenvectors[:, :2]
print("\nPrincipal Components (first 2):\n", PC)
#“The first eigenvalue is much larger, meaning the first principal component captures most variance.”

# Step 7: Project data
#Multiplies the centered data (12×6) by the projection matrix (6×2), giving a 12×2 result, so each country now has just 2 coordinates in the new PCA space.
X_pca = Xc @ PC
print("\nProjected Data (2D):\n", X_pca)

# Step 8: Scatter plot
#Plots each country as a dot on a 2D chart. Countries that are close together have similar socioeconomic profiles; countries far apart are very different.
plt.scatter(X_pca[:,0], X_pca[:,1])
for i, label in enumerate(["A","B","C","D","E","F","G","H","I","J","K","L"]):
    plt.text(X_pca[i,0], X_pca[i,1], label)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Projection")
plt.show()

# Step 9: Scree plot
#Plots all 6 eigenvalues in order. Used to decide how many components to keep, so we look for the "elbow" where the curve flattens out, meaning additional components add little new information.
plt.plot(eigenvalues, marker='o')
plt.title("Scree Plot")
plt.xlabel("Component")
plt.ylabel("Eigenvalue")
plt.show()

#Raw 12×6 data  →  Center  →  Covariance Matrix  →  Eigen-decomposition→  Sort & Pick top 2  →  Project to 2D  →  Visualize


