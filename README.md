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