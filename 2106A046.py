import sys
import numpy as np
import pandas as pd
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QTabWidget, QPushButton, QLabel, 
                           QComboBox, QFileDialog, QSpinBox, QDoubleSpinBox,
                           QGroupBox, QScrollArea, QTextEdit, QStatusBar,
                           QProgressBar, QCheckBox, QGridLayout, QMessageBox,
                           QDialog, QLineEdit, QInputDialog, QListWidget,QFormLayout)
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn import datasets, preprocessing, model_selection
from sklearn.model_selection import cross_validate, train_test_split, KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss, hinge_loss, r2_score, make_scorer,f1_score,classification_report
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.models import load_model, model_from_json # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

from sklearn.impute import SimpleImputer
import umap


class MLCourseGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Machine Learning Course GUI")
        self.setGeometry(100, 100, 1400, 800)
        
        # Initialize main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)
        
        # Initialize data containers
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.current_model = None
        self.input_image_shape = None

        # Neural network configuration
        self.mlp_layer_config = []
        self.cnn_layer_config = []
        self.rnn_layer_config = []       
        self.gen_layer_config = []
        self.disc_layer_config = []
 
        # Create components
        self.create_data_section()
        self.create_tabs()
        self.create_visualization()
        self.create_status_bar()

    def load_dataset(self):
        """Load selected dataset"""
        try:
            dataset_name = self.dataset_combo.currentText()
            
            if dataset_name == "Load Custom Dataset":
                return
            
            # Load selected dataset
            if dataset_name == "Iris Dataset":
                data = datasets.load_iris()
            elif dataset_name == "Breast Cancer Dataset":
                data = datasets.load_breast_cancer()
            elif dataset_name == "Digits Dataset":
                data = datasets.load_digits()
            elif dataset_name == "California Housing Dataset":
                data = datasets.fetch_california_housing()
            elif dataset_name == "MNIST Dataset":
                (Xtr, ytr), (Xte, yte) = tf.keras.datasets.mnist.load_data()
                Xtr = Xtr.reshape(-1,28,28,1)/255.0
                Xte = Xte.reshape(-1,28,28,1)/255.0
                self.X_train, self.y_train = Xtr, ytr
                self.X_test,  self.y_test  = Xte, yte

                # → Burada:
                self.update_input_image_shape()
                self.status_bar.showMessage("Loaded MNIST Dataset")
                return

            
            self.data = data.data       # tüm örnekler, tüm özellikler
            self.targets = data.target  # tüm etiketler
            # Split data
            test_size = self.split_spin.value()
            self.X_train, self.X_test, self.y_train, self.y_test = \
                model_selection.train_test_split(data.data, data.target, 
                                              test_size=test_size, 
                                              random_state=42)


            # Apply scaling if selected
            self.apply_missing_value_strategy()
            self.apply_scaling()
            self.update_input_image_shape()
            self.status_bar.showMessage(f"Loaded {dataset_name}")
            
        except Exception as e:
            self.show_error(f"Error loading dataset: {str(e)}")
    
    def load_custom_data(self):
        """Load custom dataset from CSV file"""
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Load Dataset",
                "",
                "CSV files (*.csv)"
            )
            
            if file_name:
                # Load data
                data = pd.read_csv(file_name)
                
                # Ask user to select target column
                target_col = self.select_target_column(data.columns)
                
                if target_col:
                    X = data.drop(target_col, axis=1)
                    y = data[target_col]
                    
                    # Split data
                    test_size = self.split_spin.value()
                    self.X_train, self.X_test, self.y_train, self.y_test = \
                        model_selection.train_test_split(X, y, 
                                                      test_size=test_size, 
                                                      random_state=42)
                    
                    # Apply scaling if selected
                    self.apply_missing_value_strategy()
                    self.apply_scaling()
                    # Bu durumda numpy array'e çeviriyoruz.
                    if isinstance(self.X_train, pd.DataFrame):
                        self.X_train = self.X_train.values
                    if isinstance(self.X_test, pd.DataFrame):
                        self.X_test = self.X_test.values
                       
                    self.update_input_image_shape()
                    self.status_bar.showMessage(f"Loaded custom dataset: {file_name}")
                    
        except Exception as e:
            self.show_error(f"Error loading custom dataset: {str(e)}")
    
    def select_target_column(self, columns):
        """Dialog to select target column from dataset"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Target Column")
        layout = QVBoxLayout(dialog)
        
        combo = QComboBox()
        combo.addItems(columns)
        layout.addWidget(combo)
        
        btn = QPushButton("Select")
        btn.clicked.connect(dialog.accept)
        layout.addWidget(btn)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return combo.currentText()
        return None
    
    def apply_scaling(self):
        """Apply selected scaling method to the data"""
        scaling_method = self.scaling_combo.currentText()
        
        if scaling_method != "No Scaling":
            try:
                if scaling_method == "Standard Scaling":
                    scaler = preprocessing.StandardScaler()
                elif scaling_method == "Min-Max Scaling":
                    scaler = preprocessing.MinMaxScaler()
                elif scaling_method == "Robust Scaling":
                    scaler = preprocessing.RobustScaler()
                
                self.X_train = scaler.fit_transform(self.X_train)
                self.X_test = scaler.transform(self.X_test)
                
            except Exception as e:
                self.show_error(f"Error applying scaling: {str(e)}")
    
    def create_data_section(self):
        """Create the data loading and preprocessing section"""
        data_group = QGroupBox("Data Management")
        data_layout = QHBoxLayout()
        
        # Dataset selection
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems([
            "Load Custom Dataset",
            "Iris Dataset",
            "Breast Cancer Dataset",
            "Digits Dataset",
            "California Housing Dataset",  # updated
            "MNIST Dataset"
        ])
        self.dataset_combo.currentIndexChanged.connect(self.load_dataset)
        
        # Data loading button
        self.load_btn = QPushButton("Load Data")
        self.load_btn.clicked.connect(self.load_custom_data)
        
        # Dataset visualization button
        self.plot_btn = QPushButton("Visualize Data")
        self.plot_btn.clicked.connect(self.plot_dataset)
        
        # Preprocessing options
        self.scaling_combo = QComboBox()
        self.scaling_combo.addItems([
            "No Scaling",
            "Standard Scaling",
            "Min-Max Scaling",
            "Robust Scaling"
        ])
        
        # Train-test split options
        self.split_spin = QDoubleSpinBox()
        self.split_spin.setRange(0.01, 0.9)
        self.split_spin.setValue(0.1)
        self.split_spin.setSingleStep(0.1)

        
        
        # Eksik veri işleme seçenekleri
        self.imputation_combo = QComboBox()
        self.imputation_combo.addItems([
            "No Imputation",
            "Mean Imputation",  
            "Median Imputation",
            "Mode Imputation",
            "Interpolation",
            "Forward Fill",
            "Backward Fill"
        ])



        # Add widgets to layout
        data_layout.addWidget(QLabel("Dataset:"))
        data_layout.addWidget(self.dataset_combo)
        data_layout.addWidget(self.load_btn)
        data_layout.addWidget(self.plot_btn)
        data_layout.addWidget(QLabel("Scaling:"))
        data_layout.addWidget(self.scaling_combo)
        data_layout.addWidget(QLabel("Test Split:"))
        data_layout.addWidget(self.split_spin)
        data_layout.addWidget(QLabel("Missing Value Strategy:"))
        data_layout.addWidget(self.imputation_combo)

        
        data_group.setLayout(data_layout)
        self.layout.addWidget(data_group)
 
    def create_tabs(self):
        """Create tabs for different ML topics"""
        self.tab_widget = QTabWidget()
        
        # Create individual tabs
        tabs = [
            ("Classical ML", self.create_classical_ml_tab),
            ("Deep Learning", self.create_deep_learning_tab),
            ("Dimensionality Reduction (USL)", self.create_dim_reduction_usl_tab),
            ("Dimensionality Reduction (SL)", self.create_dim_reduction_sl_tab),
            ("Reinforcement Learning", self.create_rl_tab),
            ("Projection Analysis", self.create_projection_analysis_tab),
        ]
        
        for tab_name, create_func in tabs:
            scroll = QScrollArea()
            tab_widget = create_func()
            scroll.setWidget(tab_widget)
            scroll.setWidgetResizable(True)
            self.tab_widget.addTab(scroll, tab_name)
        
        self.layout.addWidget(self.tab_widget)
    
    def create_classical_ml_tab(self):
        """Create the classical machine learning algorithms tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # Regression section
        regression_group = QGroupBox("Regression")
        regression_layout = QVBoxLayout()
        
        # Linear Regression
        lr_group = self.create_algorithm_group(
            "Linear Regression",
            {"fit_intercept": "checkbox",
             "loss function": ["MSE (Mean Squared Error)", "MAE (Mean Absolute Error)", "Huber Loss"]}

        )
        regression_layout.addWidget(lr_group)
        
        # Logistic Regression
        logistic_group = self.create_algorithm_group(
            "Logistic Regression",
            {"C": "double",
             "max_iter": "int",
             "multi_class": ["ovr", "multinomial"],
             "loss function": ["Cross-Entropy", "Hinge Loss"]}
        )
        regression_layout.addWidget(logistic_group)
        
        regression_group.setLayout(regression_layout)
        layout.addWidget(regression_group, 0, 0)
        
        # Classification section
        classification_group = QGroupBox("Classification")
        classification_layout = QVBoxLayout()
        
        # Naive Bayes
        nb_group = self.create_algorithm_group(
            "Naive Bayes",
            {"var_smoothing": "double", "Prior Probabilities": ["Uniform", "User-defined"]}
        )

        classification_layout.addWidget(nb_group)
        
        # SVM
        svm_group = self.create_algorithm_group(
            "Support Vector Machine",
            {"Model Type": ["SVC (Classification)", "SVR (Regression)"],
             "C": "double",
             "kernel": ["linear", "rbf", "poly"],
             "degree": "int",
             "loss function for SVC": ["Cross-Entropy", "Hinge Loss"],
             "loss function for SVR": ["MSE (Mean Squared Error)", "MAE (Mean Absolute Error)", "Huber Loss"],
             "epsilon for SVR": "double"}
        )
        classification_layout.addWidget(svm_group)
        
        # Decision Trees
        dt_group = self.create_algorithm_group(
            "Decision Tree",
            {"max_depth": "int",
             "min_samples_split": "int",
             "criterion": ["gini", "entropy"]}
        )
        classification_layout.addWidget(dt_group)
        
        # Random Forest
        rf_group = self.create_algorithm_group(
            "Random Forest",
            {"n_estimators": "int",
             "max_depth": "int",
             "min_samples_split": "int"}
        )
        classification_layout.addWidget(rf_group)
        
        # KNN
        knn_group = self.create_algorithm_group(
            "K-Nearest Neighbors",
            {"n_neighbors": "int",
             "weights": ["uniform", "distance"],
             "metric": ["euclidean", "manhattan"]}
        )
        classification_layout.addWidget(knn_group)
        
        classification_group.setLayout(classification_layout)
        layout.addWidget(classification_group, 0, 1)
        
        # === Model Evaluation Bölümü ===
        eval_group = QGroupBox("Model Evaluation (Validation/Test)")
        eval_layout = QVBoxLayout()

        # Seçim: K-Fold vs. Manual Split
        self.eval_mode_combo = QComboBox()
        self.eval_mode_combo.addItems([
            "K-Fold Cross-Validation",
            "Manual Train/Val/Test Split"
        ])
        eval_layout.addWidget(QLabel("Evaluation Mode:"))
        eval_layout.addWidget(self.eval_mode_combo)
    
        # K-Fold için k değeri
        self.eval_k_spin = QSpinBox()
        self.eval_k_spin.setRange(2, 20)
        self.eval_k_spin.setValue(5)
        eval_layout.addWidget(QLabel("Number of folds (k):"))
        eval_layout.addWidget(self.eval_k_spin)
    
        # Manual split için oran
        self.eval_split_combo = QComboBox()
        self.eval_split_combo.addItems([
            "70-15-15",
            "80-10-10",
            "60-20-20"
        ])
        eval_layout.addWidget(QLabel("Train/Val/Test Ratio:"))
        eval_layout.addWidget(self.eval_split_combo)

        model_select_label = QLabel("Select Model for Evaluation:")
        self.eval_model_combo = QComboBox()
        self.eval_model_combo.addItems([
            "Linear Regression",
            "Logistic Regression",
            "Naive Bayes",
            "Support Vector Machine"
        ])
        eval_layout.addWidget(model_select_label)
        eval_layout.addWidget(self.eval_model_combo)

    
        # Çalıştırma butonu
        eval_btn = QPushButton("Run Evaluation")
        eval_btn.clicked.connect(self.run_model_evaluation)
        eval_layout.addWidget(eval_btn)
    
        eval_group.setLayout(eval_layout)
        layout.addWidget(eval_group, 1, 0, 1, 2)  # 2 sütunluk grid’de alt satır
        
    
        return widget
    
    def create_dim_reduction_usl_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # === PCA Bölümü ===
        pca_group = QGroupBox("Principal Component Analysis (PCA)")
        pca_layout = QVBoxLayout()

        # Parametre alanlarını widget olarak oluştur
        pca_params = self.create_algorithm_group(
            "PCA Parameters",
            {
                "n_components": "int",
                "whiten": "checkbox"
            }
        )
        pca_layout.addWidget(pca_params)
        pca_group.setLayout(pca_layout)


        # === t-SNE / UMAP Bölümü ===
        tsne_umap_group = QGroupBox("t-SNE / UMAP Projections")
        proj_layout = QVBoxLayout()

        self.proj_method_combo = QComboBox()
        self.proj_method_combo.addItems(["t-SNE", "UMAP"])

        self.proj_n_components = QSpinBox()
        self.proj_n_components.setRange(1, 5)
        self.proj_n_components.setValue(2)

        self.proj_perplexity = QDoubleSpinBox()
        self.proj_perplexity.setRange(5, 100)
        self.proj_perplexity.setValue(30)

        proj_train_btn = QPushButton("Run Projection")
        proj_train_btn.clicked.connect(self.run_tsne_or_umap_projection)  

        proj_layout.addWidget(QLabel("Method:"))
        proj_layout.addWidget(self.proj_method_combo)
        proj_layout.addWidget(QLabel("n_components:"))
        proj_layout.addWidget(self.proj_n_components)
        proj_layout.addWidget(QLabel("Perplexity (for t-SNE only):"))
        proj_layout.addWidget(self.proj_perplexity)
        proj_layout.addWidget(proj_train_btn)
        tsne_umap_group.setLayout(proj_layout)
        
        # === KMeans Parametrik Eğitim ===
        kmeans_train_group = QGroupBox("KMeans - Add Cluster Labels to Dataset")
        kmeans_train_layout = QVBoxLayout()

        self.kmeans_n_clusters = QSpinBox()
        self.kmeans_n_clusters.setRange(1, 20)
        self.kmeans_n_clusters.setValue(3)

        kmeans_train_btn = QPushButton("Train KMeans and Add Cluster Feature")
        kmeans_train_btn.clicked.connect(self.train_kmeans_and_add_cluster_feature)

        kmeans_train_layout.addWidget(QLabel("Number of Clusters (k):"))
        kmeans_train_layout.addWidget(self.kmeans_n_clusters)
        kmeans_train_layout.addWidget(kmeans_train_btn)
        # kmeans_train_group.setLayout(kmeans_train_layout)

        self.kmeans_max_k = QSpinBox()
        self.kmeans_max_k.setRange(1, 20)
        self.kmeans_max_k.setValue(10)

        kmeans_btn = QPushButton("Run KMeans & Show Elbow Graph")
        kmeans_btn.clicked.connect(self.run_kmeans_and_plot_elbow)  

        kmeans_train_layout.addWidget(QLabel("Max number of clusters (k):"))
        kmeans_train_layout.addWidget(self.kmeans_max_k)
        kmeans_train_layout.addWidget(kmeans_btn)

        # ---- Quality Metrics ----
        quality_btn = QPushButton("Show Clustering Quality Metrics")
        quality_btn.clicked.connect(self.show_kmeans_quality_metrics)
        kmeans_train_layout.addWidget(quality_btn)

        kmeans_train_group.setLayout(kmeans_train_layout)



        # === Ana Layout'a Ekle ===
        layout.addWidget(pca_group)
        layout.addWidget(tsne_umap_group)
        layout.addWidget(kmeans_train_group)

        return widget

    def create_dim_reduction_sl_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # === LDA Parametre Grubu ===
        lda_group = QGroupBox("Linear Discriminant Analysis (LDA)")
        lda_layout = QVBoxLayout()

        lda_params = self.create_algorithm_group(
            "LDA Parameters",
            {
                "n_components": "int"
            }
        )
        
        lda_layout.addWidget(lda_params)
        lda_group.setLayout(lda_layout)

        layout.addWidget(lda_group)

        # === Manual 1D Projection From Given Covariance ===
        manual_group = QGroupBox("Manual 1D Projection (From Covariance Matrix)")
        manual_layout = QVBoxLayout()

        manual_proj_btn = QPushButton("Project using eigenvector of Σ = [[5, 2], [2, 3]]")
        manual_proj_btn.clicked.connect(self.manual_covariance_projection)

        manual_layout.addWidget(manual_proj_btn)
        manual_group.setLayout(manual_layout)
        layout.addWidget(manual_group)
        

        return widget
    
    def create_rl_tab(self):
        """Create the reinforcement learning tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # Environment selection
        env_group = QGroupBox("Environment")
        env_layout = QVBoxLayout()
        
        self.env_combo = QComboBox()
        self.env_combo.addItems([
            "CartPole-v1",
            "MountainCar-v0",
            "Acrobot-v1"
        ])
        env_layout.addWidget(self.env_combo)
        
        env_group.setLayout(env_layout)
        layout.addWidget(env_group, 0, 0)
        
        # RL Algorithm selection
        algo_group = QGroupBox("RL Algorithm")
        algo_layout = QVBoxLayout()
        
        self.rl_algo_combo = QComboBox()
        self.rl_algo_combo.addItems([
            "Q-Learning",
            "SARSA",
            "DQN"
        ])
        algo_layout.addWidget(self.rl_algo_combo)
        
        algo_group.setLayout(algo_layout)
        layout.addWidget(algo_group, 0, 1)
        
        return widget
    
    def create_visualization(self):
        """Create the visualization section"""
        viz_group = QGroupBox("Visualization")
        viz_layout = QHBoxLayout()
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(6, 4), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        viz_layout.addWidget(self.canvas)

        # Yeni ikinci canvas (tahmin sonuçları için)
        self.prediction_figure = Figure(figsize=(6, 4))
        self.prediction_canvas = FigureCanvas(self.prediction_figure)
        viz_layout.addWidget(self.prediction_canvas)
        
        # Metrics display
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setMaximumWidth(180)
        viz_layout.addWidget(self.metrics_text)
        
        viz_group.setLayout(viz_layout)
        self.layout.addWidget(viz_group)
    
    def create_status_bar(self):
        """Create the status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Add progress bar
        self.progress_bar = QProgressBar()
        self.status_bar.addPermanentWidget(self.progress_bar)
    
    def create_algorithm_group(self, name, params):
        """Helper method to create algorithm parameter groups"""
        group = QGroupBox(name)
        layout = QVBoxLayout()

        # Erişim için bir sözlük ekleyelim
        if not hasattr(self, "widgets_dict"):
            self.widgets_dict = {}

        # Her parametre için uygun widget'ı oluştur
        param_widgets = {}
        for param_name, param_type in params.items():
            param_layout = QHBoxLayout()
            param_layout.addWidget(QLabel(f"{param_name}:"))

            if param_type == "int":
                widget = QSpinBox()
                widget.setRange(1, 1000)
            elif param_type == "double":
                widget = QDoubleSpinBox()
                widget.setRange(0.0001, 1000.0)
                widget.setSingleStep(0.1)
            elif param_type == "checkbox":
                widget = QCheckBox()
            elif isinstance(param_type, list):
                widget = QComboBox()
                widget.addItems(param_type)

            param_layout.addWidget(widget)
            param_widgets[param_name] = widget
            layout.addLayout(param_layout)

        # Oluşturulan widget'ları self.widgets_dict içine kaydet
        self.widgets_dict[name] = param_widgets

        train_function_map = {
            "Linear Regression": self.train_linear_regression,
            "Logistic Regression": self.train_logistic_regression,
            "Naive Bayes": self.train_gaussian_nb,
            "Support Vector Machine": self.train_svm,
            "PCA Parameters": self.train_pca_and_plot_variance,
            "LDA Parameters": self.train_lda_and_show_separation
        }

        # Eğitme butonu ekleyelim ve fonksiyona bağlayalım
        train_btn = QPushButton(f"Train {name}")

        # Eğer model özel bir fonksiyon içeriyorsa, onu çağır
        train_btn.clicked.connect(train_function_map.get(name, lambda: self.show_error(f"Training function not found for {name}")))

        layout.addWidget(train_btn)
        group.setLayout(layout)
        return group

    def show_error(self, message):
        """Show error message dialog"""
        QMessageBox.critical(self, "Error", message)
       
    def create_deep_learning_tab(self):
        tab = QWidget()
        layout = QGridLayout(tab)

        # ─── MLP Bölümü ────────────────────────────────────────────────
        mlp_group = QGroupBox("Multi-Layer Perceptron")
        mlp_layout = QVBoxLayout(mlp_group)

        self.mlp_layer_list = QListWidget()
        mlp_layout.addWidget(self.mlp_layer_list)

        mlp_btns = QHBoxLayout()
        mlp_add    = QPushButton("Add MLP Layer")
        mlp_delete = QPushButton("Delete Layer")
        mlp_up     = QPushButton("↑")
        mlp_down   = QPushButton("↓")
        mlp_btns.addWidget(mlp_add)
        mlp_btns.addWidget(mlp_delete)
        mlp_btns.addWidget(mlp_up)
        mlp_btns.addWidget(mlp_down)
        mlp_layout.addLayout(mlp_btns)

        mlp_add.clicked.connect(self.add_mlp_layer_dialog)
        mlp_delete.clicked.connect(self.remove_mlp_layer)
        mlp_up.clicked.connect(self.move_mlp_layer_up)
        mlp_down.clicked.connect(self.move_mlp_layer_down)

        mlp_train = QPushButton("Train MLP Network")
        mlp_train.clicked.connect(self.train_mlp_network)
        mlp_layout.addWidget(mlp_train)
        layout.addWidget(mlp_group, 0, 0)


        # ─── CNN Bölümü ────────────────────────────────────────────────
        cnn_group = QGroupBox("Convolutional Neural Network")
        cnn_layout = QVBoxLayout(cnn_group)

        self.cnn_layer_list = QListWidget()
        cnn_layout.addWidget(self.cnn_layer_list)

        cnn_btns = QHBoxLayout()
        cnn_add    = QPushButton("Add CNN Layer")
        cnn_delete = QPushButton("Delete Layer")
        cnn_up     = QPushButton("↑")
        cnn_down   = QPushButton("↓")
        cnn_btns.addWidget(cnn_add)
        cnn_btns.addWidget(cnn_delete)
        cnn_btns.addWidget(cnn_up)
        cnn_btns.addWidget(cnn_down)
        cnn_layout.addLayout(cnn_btns)

        cnn_add.clicked.connect(self.add_cnn_layer_dialog)
        cnn_delete.clicked.connect(self.remove_cnn_layer)
        cnn_up.clicked.connect(self.move_cnn_layer_up)
        cnn_down.clicked.connect(self.move_cnn_layer_down)
        cnn_train = QPushButton("Train CNN Network")
        cnn_train.clicked.connect(self.train_cnn_network)
        cnn_layout.addWidget(cnn_train)

        # Augmentation kontrollerini tanımla
        self.rot_spin = QSpinBox()
        self.rot_spin.setRange(0, 180)
        self.rot_spin.setValue(0)

        self.hflip_chk = QCheckBox("Horizontal Flip")
        self.vflip_chk = QCheckBox("Vertical Flip")

        self.zoom_spin = QDoubleSpinBox()
        self.zoom_spin.setRange(0.0, 1.0)
        self.zoom_spin.setSingleStep(0.1)
        self.zoom_spin.setValue(0.0)

        # Kontrolleri CNN layout’a ekle (addWidget ile)
        cnn_layout.addWidget(QLabel("Rotation (deg):"))
        cnn_layout.addWidget(self.rot_spin)
        cnn_layout.addWidget(self.hflip_chk)
        cnn_layout.addWidget(self.vflip_chk)
        cnn_layout.addWidget(QLabel("Zoom Range:"))
        cnn_layout.addWidget(self.zoom_spin)


        layout.addWidget(cnn_group, 0, 1)


        # ─── RNN Bölümü ────────────────────────────────────────────────
        rnn_group = QGroupBox("Recurrent Neural Network")
        rnn_layout = QVBoxLayout(rnn_group)

        self.rnn_layer_list = QListWidget()
        rnn_layout.addWidget(self.rnn_layer_list)

        rnn_btns = QHBoxLayout()
        rnn_add    = QPushButton("Add RNN Layer")
        rnn_delete = QPushButton("Delete Layer")
        rnn_up     = QPushButton("↑")
        rnn_down   = QPushButton("↓")
        rnn_btns.addWidget(rnn_add)
        rnn_btns.addWidget(rnn_delete)
        rnn_btns.addWidget(rnn_up)
        rnn_btns.addWidget(rnn_down)
        rnn_layout.addLayout(rnn_btns)

        rnn_add.clicked.connect(self.add_rnn_layer_dialog)
        rnn_delete.clicked.connect(self.remove_rnn_layer)
        rnn_up.clicked.connect(self.move_rnn_layer_up)
        rnn_down.clicked.connect(self.move_rnn_layer_down)
        rnn_train = QPushButton("Train RNN Network")
        rnn_train.clicked.connect(self.train_rnn_network)
        rnn_layout.addWidget(rnn_train)

        layout.addWidget(rnn_group, 1, 0, 1, 2)

        # ─── Save / Load Model düğmeleri ──────────────────────────
        io_btns = QHBoxLayout()
        save_btn = QPushButton("Save Model")
        load_btn = QPushButton("Load Model")
        io_btns.addWidget(save_btn)
        io_btns.addWidget(load_btn)
        layout.addLayout(io_btns, 4, 0, 1, 2)  # 4. satır, 2 sütun genişlik

        save_btn.clicked.connect(self.save_model)
        load_btn.clicked.connect(self.load_model)

          # ─── 3) Pretrained Model Grubu ──────────────────────────
        pretrained_group = QGroupBox("Transfer Learning (Pretrained Models)")
        playout = QGridLayout(pretrained_group)

        # Architecture
        playout.addWidget(QLabel("Architecture:"), 0, 0)
        self.pre_model_combo = QComboBox()
        self.pre_model_combo.addItems(["VGG16", "ResNet50"])
        playout.addWidget(self.pre_model_combo, 0, 1)

        # Include Top
        self.include_top_chk = QCheckBox("Include Top Classifier")
        playout.addWidget(self.include_top_chk, 1, 0, 1, 2)

        # Freeze layers
        playout.addWidget(QLabel("Freeze first N layers:"), 2, 0)
        self.freeze_spin = QSpinBox()
        self.freeze_spin.setRange(0, 1000)
        self.freeze_spin.setValue(0)
        playout.addWidget(self.freeze_spin, 2, 1)

        # New Dense units
        playout.addWidget(QLabel("New Dense Units:"), 3, 0)
        self.new_dense_spin = QSpinBox()
        self.new_dense_spin.setRange(0, 1024)
        self.new_dense_spin.setValue(256)
        playout.addWidget(self.new_dense_spin, 3, 1)

        # Buttons
        btn_layout = QHBoxLayout()
        self.load_pre_btn = QPushButton("Load Pretrained Model")
        self.train_pre_btn = QPushButton("Fine-Tune & Train")
        btn_layout.addWidget(self.load_pre_btn)
        btn_layout.addWidget(self.train_pre_btn)
        playout.addLayout(btn_layout, 4, 0, 1, 2)

        layout.addWidget(pretrained_group, 3, 0, 1, 2)  # satır, sütun, rowSpan, colSpan
        # Butonları metoda bağlayın
        self.load_pre_btn.clicked.connect(self.load_pretrained_model)
        self.train_pre_btn.clicked.connect(self.train_pretrained_model)

        # --- GAN Section ---
        gan_group = QGroupBox("Generative Adversarial Network (GAN)")
        gan_layout = QVBoxLayout()

        # Generator Layer List
        gen_layout = QVBoxLayout()
        gen_layout.addWidget(QLabel("Generator Architecture"))
        self.gen_layer_list = QListWidget()
        gen_layout.addWidget(self.gen_layer_list)
        self.gen_add_btn = QPushButton("Add Generator Layer")
        self.gen_add_btn.clicked.connect(self.add_generator_layer_dialog)
        gen_layout.addWidget(self.gen_add_btn)

        # Discriminator Layer List
        disc_layout = QVBoxLayout()
        disc_layout.addWidget(QLabel("Discriminator Architecture"))
        self.disc_layer_list = QListWidget()
        disc_layout.addWidget(self.disc_layer_list)
        self.disc_add_btn = QPushButton("Add Discriminator Layer")
        self.disc_add_btn.clicked.connect(self.add_discriminator_layer_dialog)
        disc_layout.addWidget(self.disc_add_btn)

        # Generator and Discriminator buttons
        self.gen_up_btn = QPushButton("↑")
        self.gen_down_btn = QPushButton("↓")
        self.gen_remove_btn = QPushButton("Remove Layer")
        self.disc_up_btn = QPushButton("↑")
        self.disc_down_btn = QPushButton("↓")
        self.disc_remove_btn = QPushButton("Remove Layer")
        gen_layout.addWidget(self.gen_up_btn)
        gen_layout.addWidget(self.gen_down_btn)
        gen_layout.addWidget(self.gen_remove_btn)
        disc_layout.addWidget(self.disc_up_btn)
        disc_layout.addWidget(self.disc_down_btn)
        disc_layout.addWidget(self.disc_remove_btn)
        # Connect buttons to methods
        
        self.gen_up_btn.clicked.connect(self.move_generator_layer_up)
        self.gen_down_btn.clicked.connect(self.move_generator_layer_down)
        self.gen_remove_btn.clicked.connect(self.remove_generator_layer)

        self.disc_up_btn.clicked.connect(self.move_discriminator_layer_up)
        self.disc_down_btn.clicked.connect(self.move_discriminator_layer_down)
        self.disc_remove_btn.clicked.connect(self.remove_discriminator_layer)


        # Combine layouts
        gan_layout.addLayout(gen_layout)
        gan_layout.addLayout(disc_layout)

        # Train GAN Button
        self.train_gan_btn = QPushButton("Train GAN")
        self.train_gan_btn.clicked.connect(self.train_gan_network)
        gan_layout.addWidget(self.train_gan_btn)

        gan_group.setLayout(gan_layout)
        layout.addWidget(gan_group)


        # ─── **Eğitim Parametreleri Grubu** ─────────────────────────────
        # Bu satırı ekleyin, böylece self.batch_size_spin vs. oluşturulur:
        # Eğitim eğrileri/prediction’dan sonra:
        self.gradient_figure = Figure(figsize=(4,2))
        self.gradient_canvas = FigureCanvas(self.gradient_figure)
        # Örneğin grid’in  kitabınıza göre 2.,0 konumuna ekleyin:
        layout.addWidget(self.gradient_canvas, 2, 0, 1, 1)

        training_group = self.create_training_params_group()
        layout.addWidget(training_group, 2, 1, 1, 1)  # satır, sütun, rowSpan, colSpan


        return tab
    
    def create_training_params_group(self):
        """Create group for neural network training parameters"""
        group = QGroupBox("Training Parameters")
        layout = QVBoxLayout()
        
        # Batch size
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 1000)
        self.batch_size_spin.setValue(32)
        batch_layout.addWidget(self.batch_size_spin)
        layout.addLayout(batch_layout)
        
        # Epochs
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("Epochs:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(10)
        epochs_layout.addWidget(self.epochs_spin)
        layout.addLayout(epochs_layout)
        
        # Learning rate
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("Learning Rate:"))
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setSingleStep(0.001)
        lr_layout.addWidget(self.lr_spin)
        layout.addLayout(lr_layout)
        
        # Optimizer Seçimi
        opt_layout = QHBoxLayout()
        opt_layout.addWidget(QLabel("Optimizer:"))
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["Adam", "SGD", "RMSprop"])
        opt_layout.addWidget(self.optimizer_combo)
        layout.addLayout(opt_layout)

        # Learning Rate Scheduler
        sched_layout = QHBoxLayout()
        sched_layout.addWidget(QLabel("LR Scheduler:"))
        self.scheduler_combo = QComboBox()
        self.scheduler_combo.addItems(["None", "Step Decay", "Exponential Decay"])
        sched_layout.addWidget(self.scheduler_combo)
        layout.addLayout(sched_layout)

        # Step Decay parametreleri
        self.step_epoch_spin = QSpinBox()
        self.step_epoch_spin.setRange(1, 1000)
        self.step_epoch_spin.setValue(10)
        step_layout = QHBoxLayout()
        step_layout.addWidget(QLabel("Step Every N Epochs:"))
        step_layout.addWidget(self.step_epoch_spin)
        layout.addLayout(step_layout)

        self.step_drop_spin = QDoubleSpinBox()
        self.step_drop_spin.setRange(0.1, 1.0)
        self.step_drop_spin.setValue(0.5)
        drop_layout = QHBoxLayout()
        drop_layout.addWidget(QLabel("Step Drop Rate:"))
        drop_layout.addWidget(self.step_drop_spin)
        layout.addLayout(drop_layout)

        # Exponential Decay parametresi
        exp_layout = QHBoxLayout()
        exp_layout.addWidget(QLabel("Exp Decay Rate:"))
        self.exp_decay_spin = QDoubleSpinBox()
        self.exp_decay_spin.setRange(0.1, 1.0)
        self.exp_decay_spin.setSingleStep(0.1)
        self.exp_decay_spin.setValue(0.96)
        exp_layout.addWidget(self.exp_decay_spin)
        layout.addLayout(exp_layout)

        # Early Stopping
        early_layout = QHBoxLayout()
        self.early_stop_check = QCheckBox("Early Stopping")
        early_layout.addWidget(self.early_stop_check)
        early_layout.addWidget(QLabel("Patience:"))
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 100)
        self.patience_spin.setValue(5)
        early_layout.addWidget(self.patience_spin)
        layout.addLayout(early_layout)

        # default log‐dir
        tb_layout = QHBoxLayout()
        self.tb_check = QCheckBox("Enable TensorBoard")
        tb_layout.addWidget(self.tb_check)
        self.tb_logdir_input = QLineEdit("logs/")      
        tb_layout.addWidget(self.tb_logdir_input)
        layout.addLayout(tb_layout)

        group.setLayout(layout)
        return group

    def create_progress_callback(self):
        """Create callback for updating progress bar during training"""
        class ProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self, progress_bar):
                super().__init__()
                self.progress_bar = progress_bar
                
            def on_epoch_end(self, epoch, logs=None):
                progress = int(((epoch + 1) / self.params['epochs']) * 100)
                self.progress_bar.setValue(progress)
                
        return ProgressCallback(self.progress_bar)
        
    def update_visualization(self, y_pred):
        """Update the visualization with current results"""

        self.prediction_figure.clear()
        ax = self.prediction_figure.add_subplot(111)

        if len(np.unique(self.y_test)) > 10:  # Regression
            ax.scatter(self.y_test, y_pred)
            ax.plot([self.y_test.min(), self.y_test.max()],
                    [self.y_test.min(), self.y_test.max()],
                    'r--', lw=2)
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            ax.set_title("Regression Prediction vs Actual")
        else:  # Classification
            if self.X_test.shape[1] > 2:
                pca = PCA(n_components=2)
                X_test_2d = pca.fit_transform(self.X_test)
                scatter = ax.scatter(X_test_2d[:, 0], X_test_2d[:, 1],
                                     c=y_pred, cmap='viridis')
                self.prediction_figure.colorbar(scatter)
            else:
                scatter = ax.scatter(self.X_test[:, 0], self.X_test[:, 1],
                                     c=y_pred, cmap='viridis')
                self.prediction_figure.colorbar(scatter)
            ax.set_title("Classification Prediction")

        self.prediction_canvas.draw()
        
    def plot_training_history(self, history):
        """Plot neural network training history"""
        self.figure.clear()
        
        # Plot training & validation accuracy
        ax1 = self.figure.add_subplot(211)
        ax1.plot(history.history['accuracy'])
        ax1.plot(history.history['val_accuracy'])
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend(['Train', 'Test'])
        
        # Plot training & validation loss
        ax2 = self.figure.add_subplot(212)
        ax2.plot(history.history['loss'])
        ax2.plot(history.history['val_loss'])
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend(['Train', 'Test'])
        
        self.figure.tight_layout()
        self.canvas.draw()
        
    # FUNCTIONS ADDED BY ME
    def plot_dataset(self):
        """Plot the loaded dataset, skipping rows with NaN."""
        if self.X_train is None or self.y_train is None:
            self.show_error("No dataset loaded! Please load a dataset.")
            return

        try:
            X = np.array(self.X_train)
            y = np.array(self.y_train)

            # y tek boyutlu değilse flatten et
            if y.ndim > 1:
                y = y.ravel()

            # Hem X hem y'de NaN olmayan satırları seç
            mask_X = ~np.isnan(X).any(axis=1)
            mask_y = ~np.isnan(y)
            mask = mask_X & mask_y

            X = X[mask]
            y = y[mask]

            self.figure.clear()
            ax = self.figure.add_subplot(111)

            self.prediction_figure.clear()
            self.prediction_canvas.draw()

            if X.shape[1] == 2:
                scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k')
                self.figure.colorbar(scatter)
                ax.set_xlabel("Feature 1")
                ax.set_ylabel("Feature 2")
                ax.set_title("Dataset Visualization")

            elif X.shape[1] > 2:
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X)
                scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolors='k')
                self.figure.colorbar(scatter)
                ax.set_xlabel("PCA Component 1")
                ax.set_ylabel("PCA Component 2")
                ax.set_title("Dataset Visualization (PCA Reduced)")

            else:
                ax.plot(X, y, "bo")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_title("Dataset Visualization")

            self.canvas.draw()

        except Exception as e:
            self.show_error(f"Error during dataset visualization: {str(e)}")

    def huber_loss(self, y_pred, y, delta=1.0):
        huber_mse = 0.5 * (y - y_pred)**2
        huber_mae = delta * (np.abs(y - y_pred) - 0.5 * delta)
        return np.where(np.abs(y - y_pred) <= delta, huber_mse, huber_mae)
    
    def train_linear_regression(self):
        """Train and evaluate Linear Regression"""
        if self.X_train is None or self.y_train is None:
            self.show_error("No dataset loaded! Please load a dataset first.")
            return

        try:
            # Dinamik olarak erişim sağlayalım
            lr_params = self.widgets_dict.get("Linear Regression", {})

            fit_intercept = lr_params["fit_intercept"].isChecked()
            selected_loss = lr_params["loss function"].currentText()

            # Modeli oluştur ve eğit
            model = LinearRegression(fit_intercept=fit_intercept)
            self.current_model = model
            model.fit(self.X_train, self.y_train)

            # Tahminleri al
            y_pred = model.predict(self.X_test)

            # Seçili loss fonksiyonuna göre hesaplama yap
            if "MSE" in selected_loss:
                loss_value = mean_squared_error(self.y_test, y_pred)
                loss_name = "Mean Squared Error"
            elif "MAE" in selected_loss:
                loss_value = np.mean(np.abs(self.y_test - y_pred))
                loss_name = "Mean Absolute Error"
            elif "Huber" in selected_loss:
                loss_value = np.mean(self.huber_loss(y_pred, self.y_test, delta=1.0))
                loss_name = "Huber Loss"

            accuracy = r2_score(self.y_test, y_pred)

            # Sonuçları GUI'ye yazdır
            self.metrics_text.setText(f"{loss_name}: {loss_value:.4f}\nR² Score: {accuracy:.4f}")
            self.update_visualization(y_pred)

        except Exception as e:
            self.show_error(f"Error training Linear Regression: {str(e)}")
            print(f"Error: {str(e)}")

    def train_logistic_regression(self):
        """Train and evaluate Logistic Regression with Cross-Entropy Loss"""
        if self.X_train is None or self.y_train is None:
            self.show_error("No dataset loaded! Please load a dataset first.")
            return

        try:
            
            # parametreleri al
            lr_params = self.widgets_dict.get("Logistic Regression", {})

            if "C" in lr_params:
                C_value = lr_params["C"].value()
            else:
                C_value = 1.0 
            if "max_iter" in lr_params:
                max_iter_value = lr_params["max_iter"].value()
            else:
                max_iter_value = 100
            if "multi_class" in lr_params:
                multi_class_value = lr_params["multi_class"].currentText()
            else:
                multi_class_value = "ovr"
            if "loss function" in lr_params:
                selected_loss = lr_params["loss function"].currentText()
            else:
                selected_loss = "Cross-Entropy"

            model = LogisticRegression(C=C_value, max_iter=max_iter_value, multi_class=multi_class_value)
            self.current_model = model
            model.fit(self.X_train, self.y_train)
            y_pred_prob = model.predict_proba(self.X_test)
            y_pred = model.predict(self.X_test)

            # Seçili loss fonksiyonuna göre hesaplama yap
            if "Cross-Entropy" in selected_loss:
                loss_value = log_loss(self.y_test, y_pred_prob)
                loss_name = "Cross-Entropy Loss"
            else: # "Hinge" 
                # Hinge Loss için etiketlerin {-1, 1} olması gerekiyor
                y_test_hinge = np.where(self.y_test == 1, 1, -1)  
                y_decision = model.decision_function(self.X_test)  # Decision scores
                loss_value = hinge_loss(y_test_hinge, y_decision)
                loss_name = "Hinge Loss"

            # Accuracy hesapla
            accuracy = accuracy_score(self.y_test, y_pred)
            self.metrics_text.setText(f"{loss_name}: {loss_value:.4f}\nAccuracy: {accuracy:.4f}")
            self.update_visualization(y_pred)

        except Exception as e:
            self.show_error(f"Error training Logistic Regression: {str(e)}")

    def train_svm(self):
        """Train and evaluate Support Vector Machine (SVM)"""
        if self.X_train is None or self.y_train is None:
            self.show_error("No dataset loaded! Please load a dataset first.")
            return

        try:
            # GUI'den parametreleri al
            svm_params = self.widgets_dict.get("Support Vector Machine", {})

            model_type = svm_params["Model Type"].currentText() if "Model Type" in svm_params else "SVC (Classification)"
            C_value = svm_params["C"].value() if "C" in svm_params else 1.0
            kernel_value = svm_params["kernel"].currentText() if "kernel" in svm_params else "rbf"
            degree_value = svm_params["degree"].value() if "degree" in svm_params else 3
            selected_loss_SVC = svm_params["loss function for SVC"].currentText() if "loss function for SVC" in svm_params else "Cross-Entropy"
            selected_loss_SVR = svm_params["loss function for SVR"].currentText() if "loss function for SVR" in svm_params else "MSE"
            epsilon_value = svm_params["epsilon"].value() if "epsilon" in svm_params else 0.1

            # Kullanıcının seçimine göre model oluştur
            if model_type == "SVR (Regression)":
                model = SVR(C=C_value, kernel=kernel_value, degree=degree_value, epsilon=epsilon_value, gamma="auto")
            else:
                model = SVC(C=C_value, kernel=kernel_value, degree=degree_value, probability=True)
            self.current_model = model

            model.fit(self.X_train, self.y_train.ravel())
            y_pred = model.predict(self.X_test)
            # Loss hesapla
            if isinstance(model, SVC):
                y_pred_prob = model.predict_proba(self.X_test)
                # Seçili loss fonksiyonuna göre hesaplama yap
                if "Cross-Entropy" in selected_loss_SVC:
                    loss_value = log_loss(self.y_test, y_pred_prob)
                    loss_name = "Cross-Entropy Loss"
                else: # "Hinge" 
                    # Hinge Loss için etiketlerin {-1, 1} olması gerekiyor
                    y_test_hinge = np.where(self.y_test == 1, 1, -1)  
                    y_decision = model.decision_function(self.X_test)  # Decision scores
                    loss_value = hinge_loss(y_test_hinge, y_decision)
                    loss_name = "Hinge Loss"
                accuracy = accuracy_score(self.y_test, y_pred)  # Regresyon için doğruluk yerine R² skoru kullanılır
            elif isinstance(model, SVR):
                if "MSE" in selected_loss_SVR:
                    loss_value = mean_squared_error(self.y_test, y_pred)
                    loss_name = "Mean Squared Error"
                elif "MAE" in selected_loss_SVR:
                    loss_value = np.mean(np.abs(self.y_test - y_pred))
                    loss_name = "Mean Absolute Error"
                elif "Huber" in selected_loss_SVR:
                    loss_value = np.mean(self.huber_loss(y_pred, self.y_test, delta=1.0))
                    loss_name = "Huber Loss"
                accuracy = r2_score(self.y_test, y_pred)

            # Sonuçları GUI'ye yazdır
            self.metrics_text.setText(f"{loss_name}: {loss_value:.4f}\nAccuracy : {accuracy:.4f}")
            self.update_visualization(y_pred)

        except Exception as e:
            self.show_error(f"Error training SVM: {str(e)}")

    def train_gaussian_nb(self):
        """Train and evaluate GaussianNB"""
        if self.X_train is None or self.y_train is None:
            self.show_error("No dataset loaded! Please load a dataset first.")
            return

        try:
            # GUI üzerinden parametreleri alın
            nb_params = self.widgets_dict.get("Naive Bayes", {})
            var_smoothing = nb_params["var_smoothing"].value()
            prior_selection = nb_params["Prior Probabilities"].currentText()

            # Eğer kullanıcı User-defined seçerse, ek input al
            if prior_selection == "User-defined":
                priors_str, ok = QInputDialog.getText(self, "Input Priors",
                                                      "Enter prior probabilities (comma separated):")
                if ok and priors_str:
                    try:
                        priors = list(map(float, priors_str.split(',')))
                    except Exception as conv_e:
                        self.show_error(f"Error parsing prior probabilities: {str(conv_e)}")
                        return
                else:
                    self.show_error("No prior probabilities entered. Using uniform distribution.")
                    priors = None
            else:
                priors = None

            #class sayısını  printle
            # print("Class Count: ", np.unique(self.y_train, return_counts=True))

            # GaussianNB modelini oluştur ve eğit
            model = GaussianNB(var_smoothing=var_smoothing, priors=priors)
            self.current_model = model
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)

            # Model başarımını hesapla ve sonuçları göster
            accuracy = accuracy_score(self.y_test, y_pred)
            self.metrics_text.setText(f"GaussianNB Accuracy: {accuracy:.4f}")

            # Görselleştirmeyi güncelle
            self.update_visualization(y_pred)

        except Exception as e:
            self.show_error(f"Error training GaussianNB: {str(e)}")

    def apply_missing_value_strategy(self):
        """Apply missing value handling strategy to the dataset and ensure no NaN remains."""
        strategy = self.imputation_combo.currentText()

        try:
            X_train_df = pd.DataFrame(self.X_train)
            X_test_df = pd.DataFrame(self.X_test)
            Y_train_df = pd.DataFrame(self.y_train)
            Y_test_df = pd.DataFrame(self.y_test)

            if strategy == "Mean Imputation":
                imputer = SimpleImputer(strategy='mean')
                X_train_df = pd.DataFrame(imputer.fit_transform(X_train_df), columns=X_train_df.columns)
                X_test_df = pd.DataFrame(imputer.transform(X_test_df), columns=X_test_df.columns)
                Y_train_df = pd.DataFrame(imputer.fit_transform(Y_train_df), columns=Y_train_df.columns)
                Y_test_df = pd.DataFrame(imputer.transform(Y_test_df), columns=Y_test_df.columns)
            elif strategy == "Median Imputation":
                imputer = SimpleImputer(strategy='median')
                X_train_df = pd.DataFrame(imputer.fit_transform(X_train_df), columns=X_train_df.columns)
                X_test_df = pd.DataFrame(imputer.transform(X_test_df), columns=X_test_df.columns)
                Y_train_df = pd.DataFrame(imputer.fit_transform(Y_train_df), columns=Y_train_df.columns)
                Y_test_df = pd.DataFrame(imputer.transform(Y_test_df), columns=Y_test_df.columns)
            elif strategy == "Mode Imputation":
                imputer = SimpleImputer(strategy='most_frequent')
                X_train_df = pd.DataFrame(imputer.fit_transform(X_train_df), columns=X_train_df.columns)
                X_test_df = pd.DataFrame(imputer.transform(X_test_df), columns=X_test_df.columns)
                Y_train_df = pd.DataFrame(imputer.fit_transform(Y_train_df), columns=Y_train_df.columns)
                Y_test_df = pd.DataFrame(imputer.transform(Y_test_df), columns=Y_test_df.columns)
            elif strategy == "Interpolation":
                X_train_df = X_train_df.interpolate().fillna(method='bfill').fillna(method='ffill')
                X_test_df = X_test_df.interpolate().fillna(method='bfill').fillna(method='ffill')
                Y_train_df = Y_train_df.interpolate().fillna(method='bfill').fillna(method='ffill')
                Y_test_df = Y_test_df.interpolate().fillna(method='bfill').fillna(method='ffill')
            elif strategy == "Forward Fill":
                X_train_df = X_train_df.fillna(method='ffill').fillna(method='bfill')
                X_test_df = X_test_df.fillna(method='ffill').fillna(method='bfill')
                Y_train_df = Y_train_df.fillna(method='ffill').fillna(method='bfill')
                Y_test_df = Y_test_df.fillna(method='ffill').fillna(method='bfill')
            elif strategy == "Backward Fill":
                X_train_df = X_train_df.fillna(method='bfill').fillna(method='ffill')
                X_test_df = X_test_df.fillna(method='bfill').fillna(method='ffill')
                Y_train_df = Y_train_df.fillna(method='bfill').fillna(method='ffill')
                Y_test_df = Y_test_df.fillna(method='bfill').fillna(method='ffill')
            elif strategy == "No Imputation":
                # Eğer 'No Imputation' seçilmişse NaN değerleri olduğu gibi bırakıyor.
                # Ancak, burada da son kontrol yaparak eksik değerleri doldurabilirsiniz.
                pass
            
            # Verileri geri ata (y_train ve y_test için numpy array)
            self.X_train = X_train_df.values
            self.X_test = X_test_df.values
            self.y_train = Y_train_df.values
            self.y_test = Y_test_df.values

        except Exception as e:
            self.show_error(f"Error during missing value processing: {str(e)}")

    def create_projection_analysis_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # === Supervised Projection Section ===
        supervised_group = QGroupBox("Supervised Projections (LDA)")
        supervised_layout = QVBoxLayout()

        self.lda_n_components = QSpinBox()
        self.lda_n_components.setRange(1, 100)
        self.lda_n_components.setValue(2)

        train_lda_btn = QPushButton("Train & Visualize LDA")
        # Bağlantı sonra yapılacak: train_lda_btn.clicked.connect(...)

        supervised_layout.addWidget(QLabel("Number of Components (LDA):"))
        supervised_layout.addWidget(self.lda_n_components)
        supervised_layout.addWidget(train_lda_btn)
        supervised_group.setLayout(supervised_layout)

        # === Unsupervised Projection Section ===
        unsupervised_group = QGroupBox("Unsupervised Projections")
        unsup_layout = QVBoxLayout()

        self.unsup_method_combo = QComboBox()
        self.unsup_method_combo.addItems(["PCA", "t-SNE", "UMAP", "KMeans"])
        self.unsup_method_combo.currentTextChanged.connect(self.update_unsupervised_params)

        self.unsup_param_area = QVBoxLayout()

        train_unsup_btn = QPushButton("Train / Visualize")
        # Bağlantı sonra yapılacak: train_unsup_btn.clicked.connect(...)

        unsup_layout.addWidget(QLabel("Select Method:"))
        unsup_layout.addWidget(self.unsup_method_combo)
        unsup_layout.addLayout(self.unsup_param_area)
        unsup_layout.addWidget(train_unsup_btn)
        unsupervised_group.setLayout(unsup_layout)

        # === Visualization and Metrics ===
        viz_group = QGroupBox("Projection Output")
        viz_layout = QHBoxLayout()

        self.projection_figure = Figure(figsize=(5, 4))
        self.projection_canvas = FigureCanvas(self.projection_figure)

        self.projection_metrics = QTextEdit()
        self.projection_metrics.setReadOnly(True)
        self.projection_metrics.setMaximumWidth(250)

        viz_layout.addWidget(self.projection_canvas)
        viz_layout.addWidget(self.projection_metrics)
        viz_group.setLayout(viz_layout)

        # === Add to Layout ===
        layout.addWidget(supervised_group)
        layout.addWidget(unsupervised_group)
        layout.addWidget(viz_group)

        return widget

    def update_unsupervised_params(self):
        # Clear previous param widgets
        for i in reversed(range(self.unsup_param_area.count())):
            widget = self.unsup_param_area.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        method = self.unsup_method_combo.currentText()
        self.unsup_params = {}  # store inputs

        if method in ["PCA", "UMAP", "t-SNE"]:
            comp_spin = QSpinBox()
            comp_spin.setRange(1, 100)
            comp_spin.setValue(2)
            self.unsup_param_area.addWidget(QLabel("n_components"))
            self.unsup_param_area.addWidget(comp_spin)
            self.unsup_params["n_components"] = comp_spin

        if method == "t-SNE":
            perp_spin = QDoubleSpinBox()
            perp_spin.setRange(5, 100)
            perp_spin.setValue(30)
            self.unsup_param_area.addWidget(QLabel("Perplexity"))
            self.unsup_param_area.addWidget(perp_spin)
            self.unsup_params["perplexity"] = perp_spin

        if method == "KMeans":
            k_spin = QSpinBox()
            k_spin.setRange(1, 50)
            k_spin.setValue(3)
            self.unsup_param_area.addWidget(QLabel("n_clusters"))
            self.unsup_param_area.addWidget(k_spin)
            self.unsup_params["n_clusters"] = k_spin

    def show_projection_window(self, fig, metrics_text=None, title="Projection Result"):
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setMinimumSize(800, 600)

        layout = QHBoxLayout(dialog)

        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

        if metrics_text:
            metrics_box = QTextEdit()
            metrics_box.setReadOnly(True)
            metrics_box.setText(metrics_text)
            metrics_box.setMaximumWidth(250)
            layout.addWidget(metrics_box)

        dialog.exec()

    def train_pca_and_plot_variance(self):
        if self.X_train is None:
            self.show_error("Please load a dataset first.")
            return

        try:
            n_components = self.widgets_dict["PCA Parameters"]["n_components"].value()
            whiten = self.widgets_dict["PCA Parameters"]["whiten"].isChecked()
            pca = PCA(n_components=n_components, whiten=whiten)

            X_train_pca = pca.fit_transform(self.X_train)
            X_test_pca = pca.transform(self.X_test)

            # Eğitime etki edecek şekilde değişkenleri güncelle
            self.X_train = X_train_pca
            self.X_test = X_test_pca
            self.pca_model = pca  # isteğe bağlı olarak saklayabilirsin

            # Varyans oranı
            explained_var = pca.explained_variance_ratio_

            # Grafik
            fig = Figure(figsize=(6, 4))
            ax = fig.add_subplot(111)
            ax.bar(range(1, len(explained_var) + 1), explained_var * 100)
            ax.set_xlabel("Component")
            ax.set_ylabel("Explained Variance (%)")
            ax.set_title("Explained Variance per PCA Component")
            ax.set_ylim(0, 100)

            # Metrik metni
            metrics = f"PCA applied to training & test data.\n\nExplained Variance Ratio (Top {n_components}):\n"
            metrics += "\n".join([f"Component {i+1}: {v:.2%}" for i, v in enumerate(explained_var)])

            # Yeni pencerede göster
            self.show_projection_window(fig, metrics, title="PCA Result")

            # Status bar güncelle
            self.status_bar.showMessage(f"PCA applied with {n_components} components")

        except Exception as e:
            self.show_error(f"PCA failed: {str(e)}")

    def run_kmeans_and_plot_elbow(self):
        if self.X_train is None:
            self.show_error("Please load a dataset first.")
            return

        try:
            
            max_k = self.kmeans_max_k.value()
            wcss = []

            for k in range(1, max_k + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
                kmeans.fit(self.X_train)
                wcss.append(kmeans.inertia_)  # inertia_ = WCSS

            # self.last_kmeans, self.last_X ve self.last_y'nin setlendiğinden emin olunmalı
            self.last_kmeans = kmeans
            self.last_X = self.X_train

            # Elbow grafiği çiz
            fig = Figure(figsize=(6, 4))
            ax = fig.add_subplot(111)
            ax.plot(range(1, max_k + 1), wcss, marker='o')
            ax.set_title("Elbow Method - KMeans Clustering")
            ax.set_xlabel("Number of Clusters (k)")
            ax.set_ylabel("WCSS (Inertia)")
            ax.grid(True)

            # Metrik metni
            metrics = "WCSS values:\n"
            metrics += "\n".join([f"k = {k}: {w:.2f}" for k, w in zip(range(1, max_k + 1), wcss)])

            self.show_projection_window(fig, metrics, title="KMeans Elbow Method")

            self.status_bar.showMessage(f"KMeans Elbow graph generated for k = 1 to {max_k}")

        except Exception as e:
            self.show_error(f"KMeans Elbow computation failed: {str(e)}")

    def run_tsne_or_umap_projection(self):
        if self.X_train is None or self.y_train is None:
            self.show_error("Please load a dataset first.")
            return

        try:
            method = self.proj_method_combo.currentText()
            n_components = self.proj_n_components.value()
            perplexity = self.proj_perplexity.value()

            if method == "t-SNE":
                model = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
            elif method == "UMAP":
                model = umap.UMAP(n_components=n_components, random_state=42)
            else:
                self.show_error("Unknown projection method.")
                return

            # Dönüşümü uygula
            X_proj = model.fit_transform(self.X_train)

            # Görsel oluştur
            fig = Figure(figsize=(6, 5))
            if n_components == 3:
                ax = fig.add_subplot(111, projection='3d')
                scatter = ax.scatter(
                    X_proj[:, 0], X_proj[:, 1], X_proj[:, 2],
                    c=self.y_train, cmap='viridis', edgecolor='k', s=50
                )
                ax.set_xlabel("Component 1")
                ax.set_ylabel("Component 2")
                ax.set_zlabel("Component 3")
            else:
                ax = fig.add_subplot(111)
                scatter = ax.scatter(
                    X_proj[:, 0], X_proj[:, 1],
                    c=self.y_train, cmap='viridis', edgecolor='k', s=50
                )
                ax.set_xlabel("Component 1")
                ax.set_ylabel("Component 2")
                fig.colorbar(scatter, ax=ax)

            ax.set_title(f"{method} Projection ({n_components}D)")
            self.show_projection_window(fig, title=f"{method} Projection")

            self.status_bar.showMessage(f"{method} projection completed.")

        except Exception as e:
            self.show_error(f"{method} projection failed: {str(e)}")

    def train_kmeans_and_add_cluster_feature(self):
        if self.X_train is None:
            self.show_error("Please load a dataset first.")
            return

        try:
            k = self.kmeans_n_clusters.value()

            # KMeans modelini eğit
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            cluster_labels_train = kmeans.fit_predict(self.X_train)
            cluster_labels_test = kmeans.predict(self.X_test)

            # Cluster ID'leri yeni feature olarak veri setine ekle
            self.X_train = np.column_stack((self.X_train, cluster_labels_train))
            self.X_test = np.column_stack((self.X_test, cluster_labels_test))

            # Bilgi mesajı
            self.status_bar.showMessage(f"KMeans clustering applied with k = {k}. Cluster labels added to training and test data.")

            # (İsteğe bağlı: kullanıcıya metin kutusu ile gösterim)
            QMessageBox.information(self, "KMeans Completed", f"Cluster labels (k = {k}) have been added as a new feature.")

        except Exception as e:
            self.show_error(f"KMeans clustering failed: {str(e)}")

    def train_lda_and_show_separation(self):
        if self.X_train is None or self.y_train is None:
            self.show_error("Please load a labeled dataset first.")
            return
        try:
            n_components = self.widgets_dict["LDA Parameters"]["n_components"].value()
            # LDA modeli uygula
            lda = LinearDiscriminantAnalysis(n_components=n_components)
            X_train_lda = lda.fit_transform(self.X_train, self.y_train)
            X_test_lda = lda.transform(self.X_test)
            # ✅ Eğitim için verileri güncelle
            self.X_train = X_train_lda
            self.X_test = X_test_lda
            self.lda_model = lda  # İsteğe bağlı: ileride kullanmak istersen
            # Scatter plot
            fig = Figure(figsize=(6, 5))
            if n_components == 3:
                ax = fig.add_subplot(111, projection='3d')
                scatter = ax.scatter(
                    X_train_lda[:, 0], X_train_lda[:, 1], X_train_lda[:, 2],
                    c=self.y_train, cmap='viridis', edgecolor='k', s=50
                )
                ax.set_xlabel("LD1")
                ax.set_ylabel("LD2")
                ax.set_zlabel("LD3")
            else:
                ax = fig.add_subplot(111)
                scatter = ax.scatter(
                    X_train_lda[:, 0],
                    X_train_lda[:, 1] if n_components > 1 else np.zeros_like(X_train_lda[:, 0]),
                    c=self.y_train, cmap='viridis', edgecolor='k', s=50
                )
                ax.set_xlabel("LD1")
                if n_components > 1:
                    ax.set_ylabel("LD2")
                fig.colorbar(scatter, ax=ax)
            ax.set_title(f"LDA Projection ({n_components}D)")
            # Metin: Ayrım oranı
            metrics = f"LDA projection has been applied to training & test data.\n\n"
            metrics += f"Explained Variance Ratio:\n"
            metrics += "\n".join([f"LD{i+1}: {v:.2%}" for i, v in enumerate(lda.explained_variance_ratio_)])
            self.show_projection_window(fig, metrics, title="LDA Result")
            self.status_bar.showMessage(f"LDA projection applied and data updated for training.")
        except Exception as e:
            self.show_error(f"LDA failed: {str(e)}")

    def manual_covariance_projection(self):
        if self.X_train is None:
            self.show_error("Please load a dataset first.")
            return

        try:
            # Sabit kovaryans matrisi
            Sigma = np.array([[5, 2],
                              [2, 3]])

            # Özdeğerler ve özvektörler
            eigvals, eigvecs = np.linalg.eig(Sigma)
            principal_vector = eigvecs[:, np.argmax(eigvals)]

            # Sadece ilk iki özniteliği kullanarak projeksiyon yapıyoruz (X_train[:, :2])
            X_proj = self.X_train[:, :2] @ principal_vector.reshape(-1, 1)

            # X_train ve X_test'i projekte et
            self.X_train = self.X_train[:, :2] @ principal_vector.reshape(-1, 1)
            self.X_test = self.X_test[:, :2] @ principal_vector.reshape(-1, 1)

            # Grafik
            fig = Figure(figsize=(6, 4))
            ax = fig.add_subplot(111)
            ax.scatter(X_proj[:, 0], np.zeros_like(X_proj), c=self.y_train, cmap='viridis', edgecolor='k')
            ax.set_title("1D Projection using Σ Eigenvector")
            ax.set_xlabel("Projected Coordinate")
            ax.get_yaxis().set_visible(False)

            # Metin
            metrics = "Manual projection using dominant eigenvector of Σ = [[5, 2], [2, 3]]\n"
            metrics += f"Principal eigenvector: [{principal_vector[0]:.3f}, {principal_vector[1]:.3f}]\n"
            metrics += f"Corresponding eigenvalue: {eigvals[np.argmax(eigvals)]:.3f}\n"
            metrics += "\nData has been projected to 1D and stored in self.X_train/self.X_test."

            self.show_projection_window(fig, metrics, title="Manual 1D Projection")

            self.status_bar.showMessage("Manual projection applied using fixed covariance matrix.")

        except Exception as e:
            self.show_error(f"Manual projection failed: {str(e)}")

    def run_model_evaluation(self):
        if not hasattr(self, "data") or self.data is None:
            self.show_error("Please load a dataset first.")
            return

        # Özellik ve hedef
        X, y = self.data, self.targets.ravel()

        # Seçilen model
        model_name = self.eval_model_combo.currentText()
        if model_name == "Linear Regression":
            model = LinearRegression(fit_intercept=True)
        elif model_name == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_name == "Naive Bayes":
            model = GaussianNB()
        elif model_name == "Support Vector Machine":
            model = SVC(probability=True)
        else:
            self.show_error("Please select a valid model first.")
            return


        mode = self.eval_mode_combo.currentText()
        msg = ""

        # --- K-Fold CV ---
        if mode == "K-Fold Cross-Validation":
            k = self.eval_k_spin.value()
            scoring = {
                "accuracy": "accuracy",
                "mse": "neg_mean_squared_error",
                "rmse": make_scorer(lambda yt, yp: mean_squared_error(yt, yp))
            }
            results = cross_validate(model, X, y, cv=k, scoring=scoring)
            msg += f"{k}-Fold CV Results:\n"
            for met, score in scoring.items():
                vals = results[f"test_{met}"]
                if met == "mse": vals = -vals
                msg += f"{met.upper():6}: {vals.mean():.4f} ± {vals.std():.4f}\n"

        # --- Manual Split ---
        else:
            train_pct, val_pct, test_pct = map(int, self.eval_split_combo.currentText().split("-"))
            ts = (val_pct + test_pct) / 100
            X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=ts, random_state=42)
            val_ratio = val_pct / (val_pct + test_pct)
            X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio, random_state=42)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            mse = mean_squared_error(y_val, y_pred)
            rmse = mean_squared_error(y_val, y_pred)
            msg += f"Split {train_pct}-{val_pct}-{test_pct} Results:\n"
            msg += f"ACCURACY: {acc:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\n"

        self.metrics_text.setText(msg)
        self.status_bar.showMessage("Evaluation completed.")

    def show_kmeans_quality_metrics(self):
        # self.last_kmeans, self.last_X ve self.last_y'nin setlendiğinden emin olun!
        if not hasattr(self, "last_kmeans"):
            self.show_error("Please run KMeans first (elbow or training).")
            return

        X = self.last_X
        labels = self.last_kmeans.labels_

        from sklearn.metrics import (
            silhouette_score,
            calinski_harabasz_score,
            davies_bouldin_score
        )

        sil = silhouette_score(X, labels)
        ch  = calinski_harabasz_score(X, labels)
        db  = davies_bouldin_score(X, labels)

        msg = (
            f"Clustering Quality Metrics (k = {self.last_kmeans.n_clusters}):\n\n"
            f"Silhouette Score:       {sil:.4f}\n"
            f"Calinski-Harabasz Index:{ch:.4f}\n"
            f"Davies-Bouldin Score:   {db:.4f}"
        )
        QMessageBox.information(self, "KMeans Quality Metrics", msg)

    def remove_selected_layer(self):
        idx = self.layer_list.currentRow()
        if idx >= 0:
            self.layer_list.takeItem(idx)
            self.layer_config.pop(idx)

    def move_layer_up(self):
        idx = self.layer_list.currentRow()
        if idx > 0:
            item = self.layer_list.takeItem(idx)
            self.layer_list.insertItem(idx-1, item)
            # config’i eş zamanlı güncelle
            self.layer_config.insert(idx-1, self.layer_config.pop(idx))
            self.layer_list.setCurrentRow(idx-1)

    def move_layer_down(self):
        idx = self.layer_list.currentRow()
        if idx < self.layer_list.count() - 1:
            item = self.layer_list.takeItem(idx)
            self.layer_list.insertItem(idx+1, item)
            self.layer_config.insert(idx+1, self.layer_config.pop(idx))
            self.layer_list.setCurrentRow(idx+1)

    def add_mlp_layer_dialog(self):
            """Dialog to add Dense / Dropout / Flatten layer"""
            dialog = QDialog(self)
            dialog.setWindowTitle("Add MLP Layer")
            layout = QVBoxLayout(dialog)

            # Layer türü
            hl = QHBoxLayout()
            hl.addWidget(QLabel("Layer Type:"))
            type_combo = QComboBox()
            type_combo.addItems(["Dense", "Dropout", "Flatten"])
            hl.addWidget(type_combo)
            layout.addLayout(hl)

            # Parametre alanı
            params_form = QFormLayout()
            layout.addLayout(params_form)
            self.mlp_param_inputs = {}

            def update_params():
                # temizle
                while params_form.rowCount():
                    params_form.removeRow(0)
                self.mlp_param_inputs.clear()

                lt = type_combo.currentText()
                if lt == "Dense":
                    # Units
                    sb = QSpinBox()
                    sb.setRange(1, 1024)
                    sb.setValue(32)
                    params_form.addRow("Units:", sb)
                    self.mlp_param_inputs["units"] = sb

                    # Activation
                    cb = QComboBox()
                    cb.addItems(["relu", "sigmoid", "tanh"])
                    params_form.addRow("Activation:", cb)
                    self.mlp_param_inputs["activation"] = cb

                    # L2 Regularization
                    l2_sb = QDoubleSpinBox()
                    l2_sb.setRange(0.0, 1.0)
                    l2_sb.setSingleStep(0.001)
                    l2_sb.setValue(0.0)
                    params_form.addRow("L2 (λ):", l2_sb)
                    self.mlp_param_inputs["l2"] = l2_sb

                elif lt == "Dropout":
                    dsb = QDoubleSpinBox(); dsb.setRange(0,1); dsb.setSingleStep(0.1); dsb.setValue(0.5)
                    params_form.addRow("Rate:", dsb)
                    self.mlp_param_inputs["rate"] = dsb

                elif lt == "Flatten":
                    params_form.addRow(QLabel("No parameters"))

            type_combo.currentTextChanged.connect(update_params)
            update_params()

            # Butonlar
            bl = QHBoxLayout()
            ok     = QPushButton("Add Layer")
            cancel = QPushButton("Cancel")
            bl.addWidget(ok); bl.addWidget(cancel)
            layout.addLayout(bl)

            def on_add():
                lt = type_combo.currentText()
                p = {}
                for k,w in self.mlp_param_inputs.items():
                    if isinstance(w, (QSpinBox, QDoubleSpinBox)):
                        p[k] = w.value()
                    elif isinstance(w, QComboBox):
                        p[k] = w.currentText()
                # kayıt
                self.mlp_layer_config.append({"type": lt, "params": p})
                desc = f"{lt}(" + ", ".join(f"{k}={v}" for k,v in p.items()) + ")"
                self.mlp_layer_list.addItem(desc)
                dialog.accept()

            ok.clicked.connect(on_add)
            cancel.clicked.connect(dialog.reject)
            dialog.exec()

    def add_cnn_layer_dialog(self):
            """Dialog to add Conv2D / MaxPooling2D / Flatten / Dropout layer"""
            dialog = QDialog(self)
            dialog.setWindowTitle("Add CNN Layer")
            layout = QVBoxLayout(dialog)

            # Layer türü
            hl = QHBoxLayout()
            hl.addWidget(QLabel("Layer Type:"))
            type_combo = QComboBox()
            type_combo.addItems(["Conv2D", "MaxPooling2D", "Flatten", "Dropout"])
            hl.addWidget(type_combo)
            layout.addLayout(hl)

            # Parametre alanı
            params_form = QFormLayout()
            layout.addLayout(params_form)
            self.cnn_param_inputs = {}

            def update_params():
                # temizle
                while params_form.rowCount():
                    params_form.removeRow(0)
                self.cnn_param_inputs.clear()

                lt = type_combo.currentText()
                if lt == "Conv2D":
                    f = QSpinBox(); f.setRange(1,512); f.setValue(32)
                    k = QLineEdit("3,3")
                    a = QComboBox(); a.addItems(["relu","sigmoid","tanh"])
                    p = QComboBox(); p.addItems(["same","valid"])
                    params_form.addRow("Filters:", f)
                    params_form.addRow("Kernel Size:", k)
                    params_form.addRow("Activation:", a)
                    params_form.addRow("Padding:", p)
                    self.cnn_param_inputs.update(filters=f, kernel_size=k, activation=a, padding=p)

                elif lt == "MaxPooling2D":
                    pl = QLineEdit("2,2")
                    params_form.addRow("Pool Size:", pl)
                    self.cnn_param_inputs["pool_size"] = pl

                elif lt == "Flatten":
                    params_form.addRow(QLabel("No parameters"))

                elif lt == "Dropout":
                    dsb = QDoubleSpinBox(); dsb.setRange(0,1); dsb.setSingleStep(0.1); dsb.setValue(0.5)
                    params_form.addRow("Rate:", dsb)
                    self.cnn_param_inputs["rate"] = dsb

            type_combo.currentTextChanged.connect(update_params)
            update_params()

            # Butonlar
            bl = QHBoxLayout()
            ok     = QPushButton("Add Layer")
            cancel = QPushButton("Cancel")
            bl.addWidget(ok); bl.addWidget(cancel)
            layout.addLayout(bl)

            def on_add():
                lt = type_combo.currentText()
                p = {}
                for k,w in self.cnn_param_inputs.items():
                    if isinstance(w, (QSpinBox, QDoubleSpinBox)):
                        p[k] = w.value()
                    elif isinstance(w, QComboBox):
                        p[k] = w.currentText()
                    elif isinstance(w, QLineEdit):
                        txt = w.text().replace(" ","")
                        p[k] = tuple(map(int, txt.split(",")))
                # kayıt
                self.cnn_layer_config.append({"type": lt, "params": p})
                desc = f"{lt}(" + ", ".join(f"{k}={v}" for k,v in p.items()) + ")"
                self.cnn_layer_list.addItem(desc)
                dialog.accept()

            ok.clicked.connect(on_add)
            cancel.clicked.connect(dialog.reject)
            dialog.exec()

    def add_rnn_layer_dialog(self):
            """Dialog to add LSTM / GRU / Dropout / Flatten layer"""
            dialog = QDialog(self)
            dialog.setWindowTitle("Add RNN Layer")
            layout = QVBoxLayout(dialog)

            # Layer türü
            hl = QHBoxLayout()
            hl.addWidget(QLabel("Layer Type:"))
            type_combo = QComboBox()
            type_combo.addItems(["LSTM", "GRU", "Dropout", "Flatten","Dense"])
            hl.addWidget(type_combo)
            layout.addLayout(hl)

            # Parametre alanı
            params_form = QFormLayout()
            layout.addLayout(params_form)
            self.rnn_param_inputs = {}

            def update_params():
                # temizle
                while params_form.rowCount():
                    params_form.removeRow(0)
                self.rnn_param_inputs.clear()

                lt = type_combo.currentText()
                if lt in ("LSTM","GRU"):
                    sb = QSpinBox(); sb.setRange(1,512); sb.setValue(64)
                    act = QComboBox(); act.addItems(["tanh","relu","sigmoid"])
                    rs = QComboBox(); rs.addItems(["False","True"])
                    params_form.addRow("Units:", sb)
                    params_form.addRow("Activation:", act)
                    params_form.addRow("Return Seq:", rs)
                    self.rnn_param_inputs.update(units=sb, activation=act, return_sequences=rs)

                elif lt == "Dropout":
                    dsb = QDoubleSpinBox(); dsb.setRange(0,1); dsb.setSingleStep(0.1); dsb.setValue(0.5)
                    params_form.addRow("Rate:", dsb)
                    self.rnn_param_inputs["rate"] = dsb

                elif lt == "Flatten":
                    params_form.addRow(QLabel("No parameters"))

                elif lt == "Dense":
                    sb = QSpinBox(); sb.setRange(1,512); sb.setValue(1)
                    params_form.addRow("Units:", sb)              
                    act = QComboBox(); act.addItems(["linear","relu","sigmoid","tanh"])
                    params_form.addRow("Activation:", act)
                    self.rnn_param_inputs.update(units=sb, activation=act)

            type_combo.currentTextChanged.connect(update_params)
            update_params()

            # Butonlar
            bl = QHBoxLayout()
            ok     = QPushButton("Add Layer")
            cancel = QPushButton("Cancel")
            bl.addWidget(ok); bl.addWidget(cancel)
            layout.addLayout(bl)

            def on_add():
                lt = type_combo.currentText()
                p = {}
                for k,w in self.rnn_param_inputs.items():
                    if isinstance(w, (QSpinBox, QDoubleSpinBox)):
                        p[k] = w.value()
                    elif isinstance(w, QComboBox):
                        txt = w.currentText()
                        # return_sequences için bool çevir
                        if k=="return_sequences":
                            p[k] = True if txt=="True" else False
                        else:
                            p[k] = txt
                # kayıt
                self.rnn_layer_config.append({"type": lt, "params": p})
                desc = f"{lt}(" + ", ".join(f"{k}={v}" for k,v in p.items()) + ")"
                self.rnn_layer_list.addItem(desc)
                dialog.accept()

            ok.clicked.connect(on_add)
            cancel.clicked.connect(dialog.reject)
            dialog.exec()

        # Genel yardımcı: bölüm ve widget index’ini layer_config index’ine çevirir

    def add_generator_layer_dialog(self):
        self._open_gan_layer_dialog(target="generator")
    
    def add_discriminator_layer_dialog(self):
        self._open_gan_layer_dialog(target="discriminator")

    def _open_gan_layer_dialog(self, target):
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Add {'Generator' if target=='generator' else 'Discriminator'} Layer")
        layout = QVBoxLayout(dialog)

        # Layer type selection
        type_layout = QHBoxLayout()
        type_label = QLabel("Layer Type:")
        type_combo = QComboBox()
        type_combo.addItems(["Dense", "Conv2D", "Flatten", "Dropout"])
        type_layout.addWidget(type_label)
        type_layout.addWidget(type_combo)
        layout.addLayout(type_layout)

        # Parameters input
        params_group = QGroupBox("Layer Parameters")
        params_form = QFormLayout()
        params_group.setLayout(params_form)
        layout.addWidget(params_group)

        self.layer_param_inputs = {}

        def update_params():
            while params_form.rowCount():
                params_form.removeRow(0)
            self.layer_param_inputs.clear()

            lt = type_combo.currentText()
            if lt == "Dense":
                sb = QSpinBox(); sb.setRange(1, 1024); sb.setValue(32)
                params_form.addRow("Units:", sb)
                self.layer_param_inputs["units"] = sb

                act_cb = QComboBox(); act_cb.addItems(["relu", "sigmoid", "tanh", "linear"])
                params_form.addRow("Activation:", act_cb)
                self.layer_param_inputs["activation"] = act_cb

            elif lt == "Conv2D":
                filt_sb = QSpinBox(); filt_sb.setRange(1, 512); filt_sb.setValue(32)
                params_form.addRow("Filters:", filt_sb)
                self.layer_param_inputs["filters"] = filt_sb

                kern_le = QLineEdit("3,3")
                params_form.addRow("Kernel Size:", kern_le)
                self.layer_param_inputs["kernel_size"] = kern_le

                act_cb = QComboBox(); act_cb.addItems(["relu", "sigmoid", "tanh", "linear"])
                params_form.addRow("Activation:", act_cb)
                self.layer_param_inputs["activation"] = act_cb

                pad_cb = QComboBox(); pad_cb.addItems(["same", "valid"])
                params_form.addRow("Padding:", pad_cb)
                self.layer_param_inputs["padding"] = pad_cb

            elif lt == "Dropout":
                rate_sb = QDoubleSpinBox(); rate_sb.setRange(0.0, 1.0); rate_sb.setSingleStep(0.05); rate_sb.setValue(0.5)
                params_form.addRow("Rate:", rate_sb)
                self.layer_param_inputs["rate"] = rate_sb

            elif lt == "Flatten":
                params_form.addRow(QLabel("No parameters"))

        type_combo.currentIndexChanged.connect(update_params)
        update_params()

        # Buttons
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add Layer")
        cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        def add_layer():
            layer_type = type_combo.currentText()
            layer_params = {}

            for k, w in self.layer_param_inputs.items():
                if isinstance(w, QSpinBox):
                    layer_params[k] = w.value()
                elif isinstance(w, QDoubleSpinBox):
                    layer_params[k] = w.value()
                elif isinstance(w, QComboBox):
                    layer_params[k] = w.currentText()
                elif isinstance(w, QLineEdit):
                    if k == "kernel_size":
                        layer_params[k] = tuple(map(int, w.text().split(',')))

            desc = f"{layer_type}(" + ", ".join(f"{k}={v}" for k,v in layer_params.items()) + ")"

            if target == "generator":
                self.gen_layer_config.append({"type": layer_type, "params": layer_params})
                self.gen_layer_list.addItem(desc)
            elif target == "discriminator":
                self.disc_layer_config.append({"type": layer_type, "params": layer_params})
                self.disc_layer_list.addItem(desc)

            dialog.accept()

        add_btn.clicked.connect(add_layer)
        cancel_btn.clicked.connect(dialog.reject)

        dialog.exec()

    def get_config_index(self, section, widget_idx):
        """
        section: 'mlp', 'cnn' veya 'rnn'
        widget_idx: o bölüme ait QListWidget'ta seçilen satır indeksi
        """
        # Hangi layer tipleri bu bölüme ait?
        types = {
            'mlp': ['Dense', 'Dropout', 'Flatten'],
            'cnn': ['Conv2D', 'MaxPooling2D'],
            'rnn': ['LSTM', 'GRU']
        }
        allowed = types[section]
        count = 0
        for i, layer in enumerate(self.layer_config):
            if layer['type'] in allowed:
                if count == widget_idx:
                    return i
                count += 1
        return None

    # ─── MLP remove & move ────────────────────────────────────────
    def remove_mlp_layer(self):
        idx = self.mlp_layer_list.currentRow()
        if idx < 0:
            return
        # widget’tan çıkar
        self.mlp_layer_list.takeItem(idx)
        # config listesinden de çıkar
        self.mlp_layer_config.pop(idx)

    def move_mlp_layer_up(self):
        idx = self.mlp_layer_list.currentRow()
        if idx > 0:
            # swap in list widget
            item = self.mlp_layer_list.takeItem(idx)
            self.mlp_layer_list.insertItem(idx-1, item)
            self.mlp_layer_list.setCurrentRow(idx-1)
            # swap in config
            self.mlp_layer_config[idx], self.mlp_layer_config[idx-1] = \
                self.mlp_layer_config[idx-1], self.mlp_layer_config[idx]

    def move_mlp_layer_down(self):
        idx = self.mlp_layer_list.currentRow()
        if idx < self.mlp_layer_list.count()-1:
            item = self.mlp_layer_list.takeItem(idx)
            self.mlp_layer_list.insertItem(idx+1, item)
            self.mlp_layer_list.setCurrentRow(idx+1)
            self.mlp_layer_config[idx], self.mlp_layer_config[idx+1] = \
                self.mlp_layer_config[idx+1], self.mlp_layer_config[idx]

    # ─── CNN Katman Yönetimi ───────────────────────────────────────

    def remove_cnn_layer(self):
        idx = self.cnn_layer_list.currentRow()
        if idx < 0:
            return
        # Görsel listeden kaldır
        self.cnn_layer_list.takeItem(idx)
        # Config listesinden de pop
        self.cnn_layer_config.pop(idx)

    def move_cnn_layer_up(self):
        idx = self.cnn_layer_list.currentRow()
        if idx <= 0:
            return
        # Görselde yukarı taşı
        item = self.cnn_layer_list.takeItem(idx)
        self.cnn_layer_list.insertItem(idx-1, item)
        self.cnn_layer_list.setCurrentRow(idx-1)
        # Config listesinde swap
        self.cnn_layer_config[idx], self.cnn_layer_config[idx-1] = \
            self.cnn_layer_config[idx-1], self.cnn_layer_config[idx]

    def move_cnn_layer_down(self):
        idx = self.cnn_layer_list.currentRow()
        if idx < 0 or idx >= self.cnn_layer_list.count()-1:
            return
        item = self.cnn_layer_list.takeItem(idx)
        self.cnn_layer_list.insertItem(idx+1, item)
        self.cnn_layer_list.setCurrentRow(idx+1)
        self.cnn_layer_config[idx], self.cnn_layer_config[idx+1] = \
            self.cnn_layer_config[idx+1], self.cnn_layer_config[idx]

    # ─── RNN Katman Yönetimi ───────────────────────────────────────

    def remove_rnn_layer(self):
        idx = self.rnn_layer_list.currentRow()
        if idx < 0:
            return
        # Görsel listeden kaldır
        self.rnn_layer_list.takeItem(idx)
        # Config listesinden de sil
        self.rnn_layer_config.pop(idx)

    def move_rnn_layer_up(self):
        idx = self.rnn_layer_list.currentRow()
        if idx <= 0:
            return
        item = self.rnn_layer_list.takeItem(idx)
        self.rnn_layer_list.insertItem(idx-1, item)
        self.rnn_layer_list.setCurrentRow(idx-1)
        # Config elemanlarını yer değiştir
        self.rnn_layer_config[idx], self.rnn_layer_config[idx-1] = \
            self.rnn_layer_config[idx-1], self.rnn_layer_config[idx]

    def move_rnn_layer_down(self):
        idx = self.rnn_layer_list.currentRow()
        if idx < 0 or idx >= self.rnn_layer_list.count()-1:
            return
        item = self.rnn_layer_list.takeItem(idx)
        self.rnn_layer_list.insertItem(idx+1, item)
        self.rnn_layer_list.setCurrentRow(idx+1)
        self.rnn_layer_config[idx], self.rnn_layer_config[idx+1] = \
            self.rnn_layer_config[idx+1], self.rnn_layer_config[idx]

    def move_generator_layer_up(self):
        row = self.gen_layer_list.currentRow()
        if row > 0:
            item = self.gen_layer_list.takeItem(row)
            self.gen_layer_list.insertItem(row-1, item)
            self.gen_layer_list.setCurrentRow(row-1)

    def move_generator_layer_down(self):
        row = self.gen_layer_list.currentRow()
        count = self.gen_layer_list.count()
        if 0 <= row < count-1:
            item = self.gen_layer_list.takeItem(row)
            self.gen_layer_list.insertItem(row+1, item)
            self.gen_layer_list.setCurrentRow(row+1)

    def remove_generator_layer(self):
        row = self.gen_layer_list.currentRow()
        if row >= 0:
            self.gen_layer_list.takeItem(row)

    def move_discriminator_layer_up(self):
        row = self.disc_layer_list.currentRow()
        if row > 0:
            item = self.disc_layer_list.takeItem(row)
            self.disc_layer_list.insertItem(row-1, item)
            self.disc_layer_list.setCurrentRow(row-1)

    def move_discriminator_layer_down(self):
        row = self.disc_layer_list.currentRow()
        count = self.disc_layer_list.count()
        if 0 <= row < count-1:
            item = self.disc_layer_list.takeItem(row)
            self.disc_layer_list.insertItem(row+1, item)
            self.disc_layer_list.setCurrentRow(row+1)

    def remove_discriminator_layer(self):
        row = self.disc_layer_list.currentRow()
        if row >= 0:
            self.disc_layer_list.takeItem(row)

    def create_mlp_model(self, input_shape):
        """
        Build a Sequential MLP from self.mlp_layer_config,
        correctly handling L2 regularization (kernel_regularizer).
        """
        model = models.Sequential()
        # Keras’ın uyarısını önlemek için `shape=` kullanıyoruz
        model.add(layers.InputLayer(shape=input_shape))

        # GUI'den gelen MLP katmanlarını sırayla ekle
        for layer_cfg in self.mlp_layer_config:
            layer_type = layer_cfg.get('type')
            params     = layer_cfg.get('params', {})
            
            if layer_type == 'Dense':
                units = params.get('units', 32)
                activation = params.get('activation', 'relu')
                l2_value = params.get('l2', 0.0)
                model.add(layers.Dense(units=units, 
                                    activation=activation,
                                    kernel_regularizer=regularizers.l2(l2_value)))
            elif layer_type == 'Dropout':
                rate = params.get('rate', 0.5)
                model.add(layers.Dropout(rate=rate))
            elif layer_type == 'Flatten':
                model.add(layers.Flatten())  # Çok nadiren lazım, ama ekleyelim
            # Eğer başka layer türü eklenirse, burada elif blokları ekle

        if not any(layer['type']=='Flatten' for layer in self.mlp_layer_config):
            self.mlp_layer_config.insert(0, {'type': 'Flatten', 'params': {}})
        # Son katman: sınıflandırma için softmax
        num_classes = len(np.unique(self.y_train))
        model.add(layers.Dense(num_classes, activation="softmax"))

        return model

    def create_cnn_model(self, input_shape):

        model = models.Sequential()
        added_input = False

        # 1) Determine and add the InputLayer
        if input_shape is not None:
            model.add(layers.InputLayer(input_shape=input_shape))
        if input_shape is None:
            if any(lc["type"] == "Conv2D" for lc in self.cnn_layer_config):
                input_shape = self.X_train.shape[1:]

        # 2) Add the user-configured CNN layers
        saw_flatten = False
        for layer_conf in self.cnn_layer_config:
            t, p = layer_conf["type"], layer_conf["params"]
            if t == "Conv2D":
                # 1) L2 oranını params’tan çıkar
                l2_rate = p.pop("l2", 0.0)
                # 2) Temel argümanları hazırla
                kwargs = {
                    "filters":     p["filters"],
                    "kernel_size": p["kernel_size"],
                    "activation":  p.get("activation"),
                    "padding":     p.get("padding", "same")
                }
                # 3) İlk Conv2D’ye input_shape ekle
                if not added_input:
                    kwargs["input_shape"] = input_shape
                    added_input = True
                # 4) L2 varsa kernel_regularizer ekle
                if l2_rate > 0:
                    kwargs["kernel_regularizer"] = regularizers.l2(l2_rate)
                model.add(layers.Conv2D(**kwargs))
            elif t == "MaxPooling2D":
                model.add(layers.MaxPooling2D(pool_size=p["pool_size"]))
            elif t == "Flatten":
                model.add(layers.Flatten())
                saw_flatten = True
            elif t == "Dropout":
                model.add(layers.Dropout(rate=p["rate"]))
            elif t == "Dense":
                # Opsiyonel olarak CNN içinde Dense ekleme
                l2_rate = p.pop("l2", 0.0)
                dense_kwargs = {"units": p["units"], "activation": p["activation"]}
                if l2_rate > 0:
                    dense_kwargs["kernel_regularizer"] = regularizers.l2(l2_rate)
                model.add(layers.Dense(**dense_kwargs))

        # 3) Ensure there's at least one Flatten before the final Dense
        if not saw_flatten:
            model.add(layers.Flatten())

        # 4) Final classification layer
        num_classes = len(np.unique(self.y_train))
        model.add(layers.Dense(num_classes, activation="softmax"))

        return model

    def create_rnn_model(self, input_shape):

        model = models.Sequential()
        model.add(layers.InputLayer(shape=input_shape))

        saw_flatten = False
        saw_user_dense = False

        for layer_conf in self.rnn_layer_config:
            t = layer_conf["type"]
            p = layer_conf["params"].copy()

            if t == "LSTM":
                # L2 regularization for recurrent/kernels if desired
                l2_rate = p.pop("l2", 0.0)
                lstm_kwargs = {
                    "units":             p["units"],
                    "activation":        p["activation"],
                    "return_sequences":  p.get("return_sequences", False)
                }
                if l2_rate > 0:
                    lstm_kwargs["kernel_regularizer"] = regularizers.l2(l2_rate)
                model.add(layers.LSTM(**lstm_kwargs))

            elif t == "GRU":
                l2_rate = p.pop("l2", 0.0)
                gru_kwargs = {
                    "units":             p["units"],
                    "activation":        p["activation"],
                    "return_sequences":  p.get("return_sequences", False)
                }
                if l2_rate > 0:
                    gru_kwargs["kernel_regularizer"] = regularizers.l2(l2_rate)
                model.add(layers.GRU(**gru_kwargs))

            elif t == "Dropout":
                model.add(layers.Dropout(rate=p["rate"]))

            elif t == "Flatten":
                model.add(layers.Flatten())
                saw_flatten = True

            elif t == "Dense":
                # Kullanıcının eklediği Dense katmanı
                l2_rate = p.pop("l2", 0.0)
                dense_kwargs = {"units": p["units"], "activation": p["activation"]}
                if l2_rate > 0:
                    dense_kwargs["kernel_regularizer"] = regularizers.l2(l2_rate)
                model.add(layers.Dense(**dense_kwargs))
                saw_user_dense = True

        # Eğer flatten eklenmediyse ekle
        if not saw_flatten:
            model.add(layers.Flatten())

        # Eğer kullanıcı kendi Dense eklemediyse softmax çıkış katmanı ekle
        if not saw_user_dense:
            num_classes = len(np.unique(self.y_train))
            model.add(layers.Dense(num_classes, activation="softmax"))

        return model

    def create_generator_model(self, input_shape):
        model = models.Sequential()
        model.add(layers.InputLayer(input_shape=input_shape))

        for layer_conf in self.gen_layer_config:
            t = layer_conf["type"]
            p = layer_conf["params"].copy()

            if t == "Dense":
                l2_rate = p.pop("l2", 0.0)
                units = p["units"]
                activation = p["activation"]
                dense_kwargs = {"units": units, "activation": activation}
                if l2_rate > 0:
                    dense_kwargs["kernel_regularizer"] = regularizers.l2(l2_rate)
                    model.add(layers.Dense(units=np.prod(self.X_train.shape[1:]), 
                                        activation='sigmoid'))
                    # 2) Çıktıyı (28,28,1) formatına sok
                    model.add(layers.Reshape(target_shape=self.X_train.shape[1:]))
            elif t == "Conv2D":
                l2_rate = p.pop("l2", 0.0)
                kwargs = {
                    "filters":     p["filters"],
                    "kernel_size": p["kernel_size"],
                    "activation":  p.get("activation"),
                    "padding":     p.get("padding", "same")
                }
                if l2_rate > 0:
                    kwargs["kernel_regularizer"] = regularizers.l2(l2_rate)
                model.add(layers.Conv2D(**kwargs))

            elif t == "Flatten":
                model.add(layers.Flatten())

            elif t == "Dropout":
                model.add(layers.Dropout(rate=p["rate"]))

        return model

    def create_discriminator_model(self, input_shape):
        model = models.Sequential()
        model.add(layers.InputLayer(input_shape=input_shape))

        for layer_conf in self.disc_layer_config:
            t = layer_conf["type"]
            p = layer_conf["params"].copy()

            if t == "Dense":
                l2_rate = p.pop("l2", 0.0)
                units = p["units"]
                activation = p["activation"]
                dense_kwargs = {"units": units, "activation": activation}
                if l2_rate > 0:
                    dense_kwargs["kernel_regularizer"] = regularizers.l2(l2_rate)
                model.add(layers.Dense(**dense_kwargs))

            elif t == "Conv2D":
                l2_rate = p.pop("l2", 0.0)
                kwargs = {
                    "filters":     p["filters"],
                    "kernel_size": p["kernel_size"],
                    "activation":  p.get("activation"),
                    "padding":     p.get("padding", "same")
                }
                if l2_rate > 0:
                    kwargs["kernel_regularizer"] = regularizers.l2(l2_rate)
                model.add(layers.Conv2D(**kwargs))

            elif t == "Flatten":
                model.add(layers.Flatten())

            elif t == "Dropout":
                model.add(layers.Dropout(rate=p["rate"]))

        # En sona sigmoid output katmanı ekle
        model.add(layers.Dense(1, activation="sigmoid"))
        return model
    
    def create_gan_model(self, generator, discriminator):
        # Discriminator'ı "freeze" et (yalnızca generator eğitilirken)
        discriminator.trainable = False

        model = models.Sequential()
        model.add(generator)
        model.add(discriminator)
        return model

    # ─── Ortak derleme + eğitim destek metodu ─────────────────────────────
    def _compile_and_fit(self, model, X_train, y_train, X_val, y_val, generator=None):
        self.current_model = model
        try:
            # 1) Optimizer seçimi
            lr = self.lr_spin.value()
            opt = {
                "Adam": tf.keras.optimizers.Adam(lr),
                "SGD":  tf.keras.optimizers.SGD(lr),
                "RMSprop": tf.keras.optimizers.RMSprop(lr)
            }[self.optimizer_combo.currentText()]
            # — 2) Loss/Metric ve Veri Hazırlığı —
            # Son katmanın activation'ını bul:
            last_layer = model.layers[-1]
            act = getattr(last_layer, "activation", None)
            act_name = act.__name__ if act is not None else ""

            is_regression = (act_name == "linear")
            input_shape = model.input_shape
            if len(input_shape) == 2:  # (batch_size, features)
                if X_train.ndim > 2:
                    X_train = X_train.reshape(X_train.shape[0], -1)
                    X_val   = X_val.reshape(X_val.shape[0], -1)
            
            if is_regression:
                loss_fn = "mean_squared_error"
                metrics = ["mean_absolute_error"]
                y_train_proc = y_train
                y_val_proc   = y_val
            else:
                loss_fn = "categorical_crossentropy"
                metrics = ["accuracy"]
                y_train_proc = to_categorical(y_train)
                y_val_proc   = to_categorical(y_val)
                # 2) Callbacks
            callbacks = [self.create_progress_callback()]
            if self.early_stop_check.isChecked():
                callbacks.append(tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.patience_spin.value(),
                    restore_best_weights=True
                ))
            sched = self.scheduler_combo.currentText()
            if sched == "Step Decay":
                def step_decay(epoch):
                    return lr * (self.step_drop_spin.value() ** (epoch // self.step_epoch_spin.value()))
                callbacks.append(tf.keras.callbacks.LearningRateScheduler(step_decay))
            elif sched == "Exponential Decay":
                decay = self.exp_decay_spin.value()
                callbacks.append(tf.keras.callbacks.LearningRateScheduler(lambda e: lr * (decay ** e)))

            callbacks.append(GradientHistogramCallback(self.gradient_figure, self.gradient_canvas))

            # 3) Compile
            model.compile(optimizer=opt, loss=loss_fn, metrics=metrics)

            # 4) Fit
            if generator is not None:
                # generator: datagen.flow(...) gibi bir Sequence nesnesi
                steps = len(X_train) // self.batch_size_spin.value()
                history = model.fit(
                    generator,
                    steps_per_epoch=steps,
                    epochs=self.epochs_spin.value(),
                    validation_data=(X_val, tf.keras.utils.to_categorical(y_val)),
                    callbacks=callbacks
                )
            else:
                history = model.fit(
                    X_train,
                    tf.keras.utils.to_categorical(y_train),
                    batch_size=self.batch_size_spin.value(),
                    epochs=self.epochs_spin.value(),
                    validation_data=(
                        X_val, tf.keras.utils.to_categorical(y_val)
                    ),
                    callbacks=callbacks
                )

            # 5) Görselleştir
            self.plot_training_history(history)
            self.status_bar.showMessage("Training complete")
            
            # 6) Eğitim sonrası tahmin ve görselleştirme
            y_pred = model.predict(X_val)
            self.prediction_figure.clear()
            ax = self.prediction_figure.add_subplot(111)



            if is_regression:
                # Regresyon: Gerçek vs Tahmin scatter
                ax.scatter(y_val, y_pred, alpha=0.7)
                mn, mx = min(y_val.min(), y_pred.min()), max(y_val.max(), y_pred.max())
                ax.plot([mn, mx], [mn, mx], 'r--', linewidth=2)
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.set_title("Regression: Actual vs Predicted")
                mse = mean_squared_error(y_val, y_pred)
                r2  = r2_score(y_val, y_pred)
                self.metrics_text.setText(
                    f"Final Test MSE: {mse:.4f}\n"
                    f"Final Test R²:  {r2:.4f}"
                )
            else:
                # Sınıflandırma: PCA'la 2D’ye indirip renkli scatter
                y_cls = y_pred.argmax(axis=1)
                feat = X_val.reshape(X_val.shape[0], -1) if X_val.ndim>2 else X_val
                pca = PCA(n_components=2)
                feat2 = pca.fit_transform(feat)
                scatter = ax.scatter(feat2[:,0], feat2[:,1], c=y_cls, cmap='viridis', alpha=0.7)
                cbar = self.prediction_figure.colorbar(scatter)
                cbar.set_label("Predicted Class")
                ax.set_title("Classification Prediction (PCA Embedding)")
                
                y_pred_cls = np.argmax(y_pred, axis=1)
                y_val_cls = np.asarray(y_val)            # Tensor -> NumPy array

                test_acc = accuracy_score(y_val_cls, y_pred_cls)
                test_f1  = f1_score(y_val_cls, y_pred_cls, average='macro', zero_division=0)

                # dilersen classification report da alabilirsin
                cl_report = classification_report(y_val_cls, y_pred_cls, zero_division=0)
                
                # GUI'deki metrik kutusunu güncelle
                self.metrics_text.setText(
                    f"Test Accuracy: {test_acc:.4f}\n"
                    f"Test F1-Score: {test_f1:.4f}\n\n"
                    f"Classification_report: {cl_report} \n"
                )

            # 7) Prediction canvas'ı güncelle
            # self.prediction_canvas.draw()
            # grad_data = getattr(self, 'all_gradients', None)

        except Exception as e:
            self.show_error(f"Error during training visualization:\n{e}")
            
    def train_mlp_network(self):
        if not self.mlp_layer_config:
            self.show_error("No MLP layers to train")
            return
        Xtr, Xte = self.X_train, self.X_test
        input_shape = (Xtr.shape[1],)
        # Eğer 2D tensörse (örneğin MNIST), input_shape (features,) şeklinde
        if Xtr.ndim == 2:
            input_shape = (Xtr.shape[1],)
        # Eğer 3D tensörse (örneğin zaman serisi), input_shape (timesteps, features) şeklinde
        elif Xtr.ndim == 3:
            input_shape = (Xtr.shape[1], Xtr.shape[2])
        # Eğer 4D tensörse (örneğin CNN için), input_shape (height, width, channels) şeklinde
        elif Xtr.ndim == 4:
            side = int(Xtr.shape[1]**0.5)
            Xtr = Xtr.reshape(-1, side * side)
            Xte = Xte.reshape(-1, side * side)
            input_shape = (side * side,)
        else:
            self.show_error("Unsupported X_train shape for MLP.")
            return
        # Modeli oluştur
        # Eğer input_shape (features,) şeklinde ise, MLP için uygun
        if len(input_shape) == 1:
            input_shape = (input_shape[0],)
        elif len(input_shape) == 2:
            input_shape = (input_shape[0], input_shape[1])
        else:
            self.show_error("Unsupported input shape for MLP.")
            return
        model = self.create_mlp_model(input_shape=input_shape)
        try:
            self._compile_and_fit(model, Xtr, self.y_train, Xte, self.y_test)
        except Exception as e:
            self.show_error(f"Error training MLP:\n{e}")

    def train_cnn_network(self):
        if not self.cnn_layer_config:
            self.show_error("No CNN layers to train")
            return
        Xtr, Xte = self.X_train, self.X_test

        if Xtr.ndim == 2:
            side = int(Xtr.shape[1]**0.5)
            Xtr = Xtr.reshape(-1, side, side, 1)
            Xte = Xte.reshape(-1, side, side, 1)
            input_shape = (side, side, 1)
        # if already images (e.g. MNIST 28×28×1) → keep as-is
        elif Xtr.ndim == 4:
            input_shape = Xtr.shape[1:]
        else:
            self.show_error("Unsupported X_train shape for CNN.")
            return

            # 1) DataGenerator ayarları
        datagen = ImageDataGenerator(
            rotation_range=self.rot_spin.value(),
            horizontal_flip=self.hflip_chk.isChecked(),
            vertical_flip=self.vflip_chk.isChecked(),
            zoom_range=self.zoom_spin.value()
        )
        datagen.fit(Xtr)  # eğer feature-wise normalization yapıyorsanız
        model = self.create_cnn_model(input_shape=input_shape)

       # 5) Eğitimi başlat, _compile_and_fit üzerinden
        try:
            gen = datagen.flow(
                Xtr,
                tf.keras.utils.to_categorical(self.y_train),
                batch_size=self.batch_size_spin.value()
            )
            self._compile_and_fit(
                model,
                Xtr, self.y_train,
                Xte,  self.y_test,
                generator=gen
            )
            
        except Exception as e:
            self.show_error(f"Error training CNN:\n{e}")
            
    def train_rnn_network(self):
        if not self.rnn_layer_config:
            self.show_error("No RNN layers to train")
            return

        Xtr, Xte = self.X_train, self.X_test

        # 1) CNN formatındaysa (N,28,28,1) → (N,28,28)
        if Xtr.ndim == 4:
            Xtr = np.squeeze(Xtr, axis=-1)
            Xte = np.squeeze(Xte, axis=-1)

        # 2) LSTM/GRU için 3D tensor
        if any(l["type"] in ("LSTM", "GRU") for l in self.rnn_layer_config):
            if Xtr.ndim == 2:
                # Örn: (N,784) → reshape gerekebilir (ama bu tehlikeli, her dataset için geçerli olmayabilir)
                raise ValueError("RNN için uygun shape bulunamadı: Xtr.shape = {}".format(Xtr.shape))
            input_shape = (Xtr.shape[1], Xtr.shape[2])  # (timesteps, features)
        else:
            input_shape = (Xtr.shape[1],)

        # 3) Modeli kur ve eğit
        model = self.create_rnn_model(input_shape=input_shape)

        try:
            self._compile_and_fit(model, Xtr, self.y_train, Xte, self.y_test)
        except Exception as e:
            self.show_error(f"Error training RNN:\n{e}")

    
    def create_generator_model(self, input_shape):
        """
        GUI’den gelen self.gen_layer_config’i kullanıp
        BatchNorm+LeakyReLU bloklarıyla gizli katmanları ekler,
        ardından Dense(np.prod(img_shape))+Reshape(img_shape) ile
        mutlaka 28×28×1 boyutunda bir çıktı üretir.
        """
        model = models.Sequential(name="Generator")
        model.add(layers.InputLayer(input_shape=input_shape))

        # 1) Sizin eklediğiniz katmanlar
        for cfg in self.gen_layer_config:
            # parse_layer, sizin utility fonksiyonunuz
            layer = self.parse_layer(cfg)
            model.add(layer)
            if isinstance(layer, layers.Dense):
                model.add(layers.LeakyReLU(alpha=0.2))
                model.add(layers.BatchNormalization(momentum=0.8))

        # 2) Son katman: MNIST boyutuna dönüştür
        img_shape = self.X_train.shape[1:]         # (28,28,1)
        model.add(layers.Dense(units=np.prod(img_shape),
                               activation='tanh'))
        model.add(layers.Reshape(target_shape=img_shape))

        return model

    def create_discriminator_model(self, input_shape):
        """
        GUI’den gelen self.disc_layer_config’i kullanıp
        Flatten + LeakyReLU+Dropout bloklarıyla devam eder
        ve en sonda sigmoid çıkış katmanını ekler.
        """
        model = models.Sequential(name="Discriminator")
        model.add(layers.InputLayer(input_shape=input_shape))
        model.add(layers.Flatten())

        for cfg in self.disc_layer_config:
            layer = self.parse_layer(cfg)
            model.add(layer)
            if isinstance(layer, layers.Dense):
                model.add(layers.LeakyReLU(alpha=0.2))
                model.add(layers.Dropout(0.4))

        model.add(layers.Dense(1, activation='sigmoid'))
        return model

    def train_gan_network(self):

        try:
            # --- 1) Veri hazırlığı ---
            latent_dim = 100
            X = self.X_train
            # MNIST ise (N,28,28,1); diğer türlü GUI’den yüklenince reshape vs. yapılmış
            X = (X - 0.5) * 2.0    # [0,1] → [-1,1]
            
            batch_size = self.batch_size_spin.value()
            half_batch = batch_size // 2
            epochs = self.epochs_spin.value()

            # --- 2) Modelleri oluştur & compile ---
            gen = self.create_generator_model(input_shape=(latent_dim,))
            disc = self.create_discriminator_model(input_shape=X.shape[1:])

            d_opt = optimizers.Adam(learning_rate=self.lr_spin.value(), beta_1=0.5)
            disc.compile(loss='binary_crossentropy', optimizer=d_opt, metrics=['accuracy'])
            disc.trainable = False

            gan_input = layers.Input(shape=(latent_dim,))
            gan_output = disc(gen(gan_input))
            gan = models.Model(gan_input, gan_output, name="GAN")
            gan.compile(loss='binary_crossentropy', optimizer=d_opt)

            # --- 3) Eğitim döngüsü ---
            for epoch in range(1, epochs+1):
                # a) Discriminator: gerçek vs fake
                idx   = np.random.randint(0, X.shape[0], half_batch)
                real  = X[idx]
                noise = np.random.normal(0, 1, (half_batch, latent_dim))
                fake  = gen.predict(noise, verbose=0)

                y_real = np.full((half_batch,1), 0.9)   # label smoothing
                y_fake = np.zeros((half_batch,1))
                d_loss_real = disc.train_on_batch(real, y_real)
                d_loss_fake = disc.train_on_batch(fake, y_fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # b) Generator via GAN
                noise2 = np.random.normal(0, 1, (batch_size, latent_dim))
                y_gen  = np.ones((batch_size,1))
                g_loss = gan.train_on_batch(noise2, y_gen)

                # c) GUI Güncelle
                msg = (f"Epoch {epoch}/{epochs}  "
                       f"D loss: {d_loss[0]:.3f}, acc: {100*d_loss[1]:.1f}%   "
                       f"G loss: {g_loss:.3f}")
                self.status_bar.showMessage(msg)
                self.gan_log_text.appendPlainText(msg)

            # --- 4) Eğitim sonrası örnek üret ve çiz ---
            sample_noise = np.random.normal(0,1,(16, latent_dim))
            gen_imgs = gen.predict(sample_noise)
            self._plot_generated_images(gen_imgs)

        except Exception as e:
            self.show_error(f"Error training GAN:\n{e}")

    def _plot_generated_images(self, gen_imgs):
        self.prediction_figure.clear()
        ax = self.prediction_figure.add_subplot(111)
        num = gen_imgs.shape[0]
        side = int(np.ceil(np.sqrt(num)))

        for i in range(num):
            plt = gen_imgs[i].squeeze()
            ax.imshow(plt, cmap='gray')
        ax.set_title("Generated Samples")
        self.prediction_canvas.draw()

    def save_model(self):
        if not hasattr(self, 'current_model'):
            self.show_error("No model to save. Please train or load a model first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Model",
            "",
            "Keras HDF5 (*.h5);;JSON Architecture (*.json)"
        )
        if not path:
            return

        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".h5":
                # model + ağırlıkları + optimizer bilgisi
                self.current_model.save(path)
            elif ext == ".json":
                # yalnızca mimari
                arch = self.current_model.to_json()
                with open(path, 'w') as f:
                    f.write(arch)
            else:
                # uzantı vermediyse varsayılan olarak .h5
                self.current_model.save(path + ".h5")
            self.status_bar.showMessage(f"Model saved to {path}")
        except Exception as e:
            self.show_error(f"Error saving model:\n{e}")

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Model",
            "",
            "Keras HDF5 (*.h5);;JSON Architecture (*.json)"
        )
        if not path:
            return

        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".h5":
                m = load_model(path)
            elif ext == ".json":
                with open(path, 'r') as f:
                    arch = f.read()
                m = model_from_json(arch)
                # JSON’den yüklenen model henüz derlenmedi, derleyelim:
                m.compile(optimizer='adam', loss='categorical_crossentropy')
            else:
                self.show_error("Unsupported model format.")
                return

            self.current_model = m
            self.status_bar.showMessage(f"Model loaded from {path}")
        except Exception as e:
            self.show_error(f"Error loading model:\n{e}")

    def load_pretrained_model(self):
        # 1) input_image_shape kesin var mı?
        shape = getattr(self, 'input_image_shape', None)
        if shape is None:
            self.show_error("No image data loaded. Please load MNIST/Digits or a 3-channel image dataset first.")
            return

        # 2) Kaç boyutlu?
        if len(shape) != 3:
            # Bu bir tabular veri
            self.show_error("Pretrained models require image data (H,W,C). Your data is tabular.")
            return

        h, w, c = shape
        # 3) Grayscale ise üç kanala çoğaltma kısmı
        Xtr, Xte = self.X_train, self.X_test
        h, w, c = self.input_image_shape

        if c == 1:
            import numpy as np
            Xtr = np.repeat(Xtr, 3, axis=-1)
            Xte = np.repeat(Xte, 3, axis=-1)
            c = 3

        # ─── 3.1) Eğer boyut 32×32’den küçükse 32×32’ye ölçekle ─────────
        if h < 32 or w < 32:
            import tensorflow as tf
            # TensorFlow ile yeniden boyutlandırma
            Xtr = tf.image.resize(Xtr, [32, 32]).numpy()
            Xte = tf.image.resize(Xte, [32, 32]).numpy()
            h, w = 32, 32

        # Güncellenmiş veriyi sakla
        self.X_train, self.X_test = Xtr, Xte
        self.input_image_shape = (h, w, c)

        # 4) Artık kesinlikle 3 kanallı ve en az 32×32
        if c != 3:
            self.show_error(f"Pretrained requires 3-channel images, but got {c} channels.")
            return

        if h < 32 or w < 32:
            self.show_error(f"Pretrained requires minimum 32×32, but got ({h},{w}).")
            return
        # 4) Artık c mutlaka 3 olmalı
        if c != 3:
            self.show_error(f"Pretrained requires 3-channel images, but got {c} channels.")
            return

        # 5) Seçimleri al
        arch = self.pre_model_combo.currentText()
        include_top = self.include_top_chk.isChecked()
        freeze_n = self.freeze_spin.value()
        new_units = self.new_dense_spin.value()

        try:
            # 6) Base modeli yükle
            if arch == "VGG16":
                from tensorflow.keras.applications import VGG16 as _ModelClass #type: ignore
            else:
                from tensorflow.keras.applications import ResNet50 as _ModelClass #type: ignore

            base = _ModelClass(
                weights="imagenet",
                include_top=include_top,
                input_shape=(h, w, c)
            )

            # 7) Freeze katmanları
            for layer in base.layers[:freeze_n]:
                layer.trainable = False
            for layer in base.layers[freeze_n:]:
                layer.trainable = True

            x = base.output
            x = GlobalAveragePooling2D()(x)
            if new_units > 0:
                x = Dense(new_units, activation="relu")(x)
            num_classes = len(np.unique(self.y_train))
            preds = Dense(num_classes, activation="softmax")(x)

            self.current_model = Model(inputs=base.input, outputs=preds)
            self.status_bar.showMessage(
                f"{arch} loaded, freeze first {freeze_n}, new_dense={new_units}"
            )

        except Exception as e:
            self.show_error(f"Error loading pretrained model:\n{e}")
        
    def train_pretrained_model(self):
        if not hasattr(self, "current_model"):
            self.show_error("No pretrained model loaded.")
            return
            # Ön koşul: mutlaka 3 kanallı görüntü verisi olmalı
        shape = getattr(self, 'input_image_shape', None)
        if not (isinstance(shape, tuple) and len(shape)==3 and shape[2]==3):
            self.show_error("Pretrained only works on 3-channel images (e.g. MNIST or your own 3-ch image data).")
            return

        # 1) Gerekirse reshape
        Xtr, Xte = self.X_train, self.X_test
            # Eğer tek kanallı (C=1) görüntü ise 3 kanala kopyala
        if Xtr.ndim == 4 and Xtr.shape[-1] == 1:
            Xtr = np.repeat(Xtr, 3, axis=-1)
            Xte = np.repeat(Xte, 3, axis=-1)
            # input_image_shape’ı da güncelle
            self.input_image_shape = (self.input_image_shape[0],
                                    self.input_image_shape[1],
                                    3)
        if Xtr.ndim == 2:  # flat vectors
            side = int(np.sqrt(Xtr.shape[1]))
            Xtr = Xtr.reshape(-1, side, side, 1)
            Xte = Xte.reshape(-1, side, side, 1)

        try:
            self._compile_and_fit(
                self.current_model,
                Xtr, self.y_train,
                Xte, self.y_test
            )
        except Exception as e:
            self.show_error(f"Error fine-tuning model:\n{e}")

    def update_input_image_shape(self):
        """
        Kendiliğinden tespit edip self.input_image_shape'e atar:
        - Eğer X_train 4D ise (N, H, W, C) doğrudan (H, W, C)
        - Eğer X_train 3D ise (N, H, W) → (H, W, 1)
        - Eğer X_train 2D ise (N, F) ve F kare değilse (F,) olarak bırakır;
            eğer F kare ise (sqrt(F), sqrt(F), 1) olarak ayarlar.
        """
        x = getattr(self, 'X_train', None)
        if x is None:
            self.input_image_shape = None
            return

        dims = x.ndim
        if dims == 4:
            # (N, H, W, C)
            self.input_image_shape = x.shape[1:]
        elif dims == 3:
            # (N, H, W) → tek kanallı
            h, w = x.shape[1], x.shape[2]
            self.input_image_shape = (h, w, 1)
        elif dims == 2:
            # (N, F) → eğer F kare sayıysa image; değilse vektör
            F = x.shape[1]
            side = int(F**0.5)
            if side * side == F:
                self.input_image_shape = (side, side, 1)
            else:
                # Regression/MLP için tek boyut halinde
                self.input_image_shape = (F,)
        else:
            # beklenmeyen boyut
            self.input_image_shape = None

    def launch_tensorboard(self):
        logdir = self.tb_logdir_input.text().strip() or "logs/"
        # This will start tensorboard in a background thread
        import threading, subprocess
        def run_tb():
            subprocess.run(["tensorboard", "--logdir", logdir, "--port", "6006"])
        threading.Thread(target=run_tb, daemon=True).start()
        QMessageBox.information(self, "TensorBoard",
                                f"TensorBoard started at http://localhost:6006\n"
                                f"Logs: {logdir}")
    
    def parse_layer(cfg):
        """
        Helper to convert a config dict to a Keras layer.
        Supports Dense, Conv2D, Flatten, Dropout. It returns layers
        """
        t = cfg["type"]
        p = cfg["params"].copy()
        if t == "Dense":
            l2_rate = p.pop("l2", 0.0) if "l2" in p else 0.0
            kwargs = {"units": p["units"], "activation": p.get("activation", "relu")}
            if l2_rate > 0:
                kwargs["kernel_regularizer"] = regularizers.l2(l2_rate)
            return layers.Dense(**kwargs)
        elif t == "Conv2D":
            l2_rate = p.pop("l2", 0.0) if "l2" in p else 0.0
            kwargs = {
                "filters": p["filters"],
                "kernel_size": p["kernel_size"],
                "activation": p.get("activation", "relu"),
                "padding": p.get("padding", "same")
            }
            if l2_rate > 0:
                kwargs["kernel_regularizer"] = regularizers.l2(l2_rate)
            return layers.Conv2D(**kwargs)
        elif t == "Flatten":
            return layers.Flatten()
        elif t == "Dropout":
            return layers.Dropout(rate=p["rate"])
        else:
            raise ValueError(f"Unknown layer type: {t}")

class GradientHistogramCallback(tf.keras.callbacks.Callback):
    def __init__(self, fig: Figure, canvas: FigureCanvas):
        super().__init__()
        self.fig = fig
        self.canvas = canvas
        self.weights_before = None

    def on_epoch_begin(self, epoch, logs=None):
        # Epoch başında ağırlıkları yakala
        self.weights_before = [w.numpy().copy() for w in self.model.trainable_variables]

    def on_epoch_end(self, epoch, logs=None):
        # Epoch sonunda güncellenmiş ağırlıkları al
        diffs = []
        for w_before, w in zip(self.weights_before, self.model.trainable_variables):
            w_after = w.numpy()
            # Güncelleme = eski - yeni (yaklaşık gradient * lr)
            update = w_before - w_after
            diffs.append(update.flatten())
        all_updates = np.concatenate(diffs)

        # Histogramı çiz
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.hist(all_updates, bins=50, density=True, alpha=0.7)
        ax.set_title(f"Weight Update Histogram\nEpoch {epoch+1}")
        ax.set_xlabel("Update value")
        ax.set_ylabel("Density")
        self.canvas.draw()




def main():
    """Main function to start the application"""
    app = QApplication(sys.argv)
    window = MLCourseGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
