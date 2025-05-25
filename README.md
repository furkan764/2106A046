# ⚙️ MKT3434_2025

**MKT3434 Course of Dept. Mechatronics Eng. at YTU instructed by Ertugrul Bayraktar**

---

## 🏁 Getting Started

1. Before uploading any data, **Scaling**, **Test Split**, and **Missing Value Method** should be configured after launching the GUI.  
2. Select the dataset you want to upload. If a custom dataset is to be used, first click on **"Load Custom Dataset"**, then click on **"Load Data"**.  
3. If the data is successfully loaded, a **"Loaded..."** message will appear at the bottom left.  
4. After the data is loaded, click **"Visualize Data"** to view the dataset.  
5. Once the data is visualized, set the desired parameters for the chosen method.  
6. After confirming the parameter settings, click the **"Train"** button for the selected model.  
7. If no errors occur during training, the training output will be displayed on the second screen, 
    and the **loss** and **accuracy scores** will be shown on the right side of the screen.  
8. If you want to perform another training session, you need to repeat all the steps starting from step 1.

**Note:** The **SVR (Support Vector Regression)** parameters within the SVM module must be carefully configured. Otherwise, the training may take an extremely long time or the program may enter an infinite loop and never complete.

**Note:** If the uploaded dataset contains **NaN** values and no missing value strategy is applied, a training error will occur.
**Note:** If **Scaling**, **Test Split**, and **Missing Value Method**  is changed, the dataset must be reloaded.

## 🆕 New Implementations

Since the initial version, the following features have been added and integrated into the GUI:

### 1. Data Splitting & Validation
- **Train/Validation/Test Split** options: 70–15–15, 80–10–10, 60–20–20  
- **K-Fold Cross-Validation** (user-selectable k between 2 and 20)  
- Combined **“Model Evaluation”** panel with a single dropdown to choose between manual split and k-fold  
- Automatic calculation and reporting of **accuracy**, **MSE**, and **RMSE**  

### 2. Supervised Dimensionality Reduction (SL Tab)
- **Linear Discriminant Analysis (LDA)** with user-selectable `n_components`  
- Visualization in 2D/3D scatter plots  

### 3. Unsupervised Dimensionality Reduction (USL Tab)
- **Principal Component Analysis (PCA)** with `n_components` and `whiten` options  
- **t-SNE** and **UMAP** projections with tunable `n_components` and `perplexity`  
- **Manual 1D Projection** from a fixed covariance matrix (Σ = [[5,2],[2,3]])  

### 4. Clustering Analysis
- **KMeans**:
  - “Train KMeans and Add Cluster Feature” adds cluster labels to dataset  
  - **Elbow method** (max_k configurable) with WCSS plot  
  - **Show Clustering Quality Metrics** button computing Silhouette, Calinski–Harabasz, and Davies–Bouldin scores  


## 🆕 New Implementations HW3

### CNN, RNN, and MLP Training

1. Follow the initial steps outlined in the **Getting Started** section.
2. Add the appropriate layers for your desired training model under the **Deep Learning** tab.
3. Click the **Train** button.
4. The results will be displayed on the second screen.
5. You can download the model by clicking the **Save Model** button.

### Pretrained Model

1. Under the **Deep Learning** tab, you can load a pretrained model.
2. Once loaded, you can continue training the model with your desired dataset.

**Note** Be careful about packages. You may need to download 
pip install umap
pip install umap-learn
pip install plotly


