# Predicting nature of tumor using K-Means
## Francisco A. Díaz Vergara A01204695

The data used for this project was obtained from: https://www.kaggle.com/datasets/ninjacoding/breast-cancer-wisconsin-benign-or-malignant

The projects' function is to predict whether a tumor is benign or malignant.

It uses the following features: Clump thickness, uniformity of cell size, uniformity of cell shape, marginal adhesion, single epithelial cell size, bare nuclei, bland chromatin, normal nucleoli and mitoses

Here's a brief explanation of each feature: 

Clump thickness: This refers to the thickness of the aggregated cells that are present in a tissue sample. In general, thicker clumps are associated with more severe abnormalities and a higher likelihood of cancer.

Uniformity of cell size: This refers to the degree to which cells in a tissue sample are the same size. Cells that are more uniform in size are generally considered less abnormal and less likely to be cancerous.

Uniformity of cell shape: This refers to the degree to which cells in a tissue sample have the same shape. Cells that are more uniform in shape are generally considered less abnormal and less likely to be cancerous.

Marginal adhesion: This refers to how well cells stick together along the edges of a tissue sample. Strong adhesion can be a sign of cancerous cells that are trying to spread to other parts of the body.

Single epithelial cell size: This refers to the size of individual epithelial cells, which are cells that line surfaces and cavities of the body. Abnormally large or small epithelial cells can be a sign of tissue abnormalities.

Bare nuclei: This refers to the presence or absence of nuclei that are not surrounded by cytoplasm. Bare nuclei can be a sign of more severe tissue abnormalities and a higher likelihood of cancer.

Bland chromatin: This refers to the uniformity of the chromatin in a cell's nucleus. Abnormally uniform chromatin can be a sign of cancerous cells.

Normal nucleoli: This refers to the presence or absence of nucleoli, which are structures within the nucleus that are involved in cell division. Abnormally large or numerous nucleoli can be a sign of more severe tissue abnormalities and a higher likelihood of cancer.

Mitoses: This refers to the number of mitotic figures, which are structures within a cell that are involved in cell division. A high number of mitotic figures can be a sign of more severe tissue abnormalities and a higher likelihood of cancer.

The project lets the user do predictions inputing a severity value (1-10) for each feature, one being the lowest and 10 being the highest severity value.

How to run: python project.py
