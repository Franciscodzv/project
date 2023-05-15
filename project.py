import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# read data from csv file
data = pd.read_csv('/Users/pacodiaz/Desktop/project/tumor.csv')

data = data.dropna()

# select all columns except the 'Class' column
data = data.iloc[:, :-1]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# drop the first column
data = data.drop('Sample code number', axis=1)

# get labels from the first row
labels = data.iloc[0]

# drop the first row
data = data.drop(0)






# convert data to numeric values
data = data.apply(pd.to_numeric)

# fit KMeans model with 2 clusters (benign vs malignant)
kmeans = KMeans(n_clusters=2, random_state=42).fit(data)

# define a function to get user input and make predictions
def predict_tumor():
    # get user input for tumor features with error handling
    while True:
        try:
            clump_thickness = float(input("Enter clump thickness (1-10): "))
            uniformity_of_cell_size = float(input("Enter uniformity of cell size (1-10): "))
            uniformity_of_cell_shape = float(input("Enter uniformity of cell shape (1-10): "))
            marginal_adhesion = float(input("Enter marginal adhesion (1-10): "))
            single_epithelial_cell_size = float(input("Enter single epithelial cell size (1-10): "))
            bare_nuclei = float(input("Enter bare nuclei (1-10): "))
            bland_chromatin = float(input("Enter bland chromatin (1-10): "))
            normal_nucleoli = float(input("Enter normal nucleoli (1-10): "))
            mitoses = float(input("Enter mitoses (1-10): "))

            if not all(1 <= x <= 10 for x in [clump_thickness, uniformity_of_cell_size, 
                                               uniformity_of_cell_shape, marginal_adhesion, 
                                               single_epithelial_cell_size, bare_nuclei, 
                                               bland_chromatin, normal_nucleoli, mitoses]):
                print("All values must be between 1 and 10.")
            else:
                break
        except ValueError:
            print("Invalid input. Please enter a valid number between 1 and 10.")

    # create a dataframe with the user input
    input_df = pd.DataFrame([[clump_thickness, uniformity_of_cell_size, uniformity_of_cell_shape,
                              marginal_adhesion, single_epithelial_cell_size, bare_nuclei,
                              bland_chromatin, normal_nucleoli, mitoses]], 
                            columns=data.columns)

    # make prediction with KMeans model
    prediction = kmeans.predict(input_df)
    if prediction == 0:
        print("The tumor is predicted to be benign.")
    else:
        print("The tumor is predicted to be malignant.")

    # plot the classification
    pca = PCA(n_components=2)
    pca.fit(data)
    transformed_data = pca.transform(data)
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=kmeans.labels_)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('KMeans Clustering')
    plt.show()

if __name__ == '__main__':
    # check the head of the data
    print(data.head())

    # call the predict_tumor function
    print(predict_tumor())

from sklearn.metrics import silhouette_score

# evaluate silhouette score
silhouette = silhouette_score(data, kmeans.labels_)


# evaluate inertia
inertia = kmeans.inertia_

#print inertia and silhouette score with 2 decimal places
print("Inertia: {:.2f}".format(inertia))
print("Silhouette score: {:.2f}".format(silhouette))


#Silhouette score: measures how similar an object is to its own cluster compared to other clusters. 
#A high silhouette score (closer to 1) indicates that the object is well-matched to its own cluster and poorly-matched to neighboring clusters.

kf = KFold(n_splits=5, shuffle=True, random_state=0)

# create lists to store results
inertias = []
silhouette_scores = []

# iterate over each fold
for train_index, test_index in kf.split(data):
    # get training and test data
    train_data = data.iloc[train_index]
    test_data = data.iloc[test_index]
    
    # fit KMeans model with training data
    kmeans.fit(train_data)
    
    # evaluate model on test data
    labels = kmeans.predict(test_data)
    silhouette = silhouette_score(test_data, labels)
    inertia = kmeans.inertia_
    
    # store results
    silhouette_scores.append(silhouette)
    inertias.append(inertia)

# calculate mean silhouette score and inertia over all folds
mean_silhouette = np.mean(silhouette_scores)
mean_inertia = np.mean(inertias)

# print results
print("Mean Inertia: {:.2f}".format(mean_inertia))
print("Mean Silhouette Score: {:.2f}".format(mean_silhouette))

