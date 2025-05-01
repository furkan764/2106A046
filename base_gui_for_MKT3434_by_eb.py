import sys
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QTabWidget, QPushButton, QLabel, 
                           QComboBox, QFileDialog, QSpinBox, QDoubleSpinBox,
                           QGroupBox, QScrollArea, QTextEdit, QStatusBar,
                           QProgressBar, QCheckBox, QGridLayout, QMessageBox,
                           QDialog, QLineEdit,QInputDialog)
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, mean_squared_error,mean_absolute_error, confusion_matrix, log_loss, hinge_loss, r2_score, make_scorer,silhouette_score, calinski_harabasz_score, davies_bouldin_score
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers # type: ignore
from sklearn.impute import SimpleImputer
import umap
import plotly.express as px


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
        
        # Neural network configuration
        self.layer_config = []
        
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
                (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
                self.X_train, self.X_test = X_train, X_test
                self.y_train, self.y_test = y_train, y_test
                self.status_bar.showMessage(f"Loaded {dataset_name}")
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
        """Create the deep learning tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # MLP section
        mlp_group = QGroupBox("Multi-Layer Perceptron")
        mlp_layout = QVBoxLayout()
        
        # Layer configuration
        self.layer_config = []
        layer_btn = QPushButton("Add Layer")
        layer_btn.clicked.connect(self.add_layer_dialog)
        mlp_layout.addWidget(layer_btn)
        
        # Training parameters
        training_params_group = self.create_training_params_group()
        mlp_layout.addWidget(training_params_group)
        
        # Train button
        train_btn = QPushButton("Train Neural Network")
        train_btn.clicked.connect(self.train_neural_network)
        mlp_layout.addWidget(train_btn)
        
        mlp_group.setLayout(mlp_layout)
        layout.addWidget(mlp_group, 0, 0)
        
        # CNN section
        cnn_group = QGroupBox("Convolutional Neural Network")
        cnn_layout = QVBoxLayout()
        
        # CNN architecture controls
        cnn_controls = self.create_cnn_controls()
        cnn_layout.addWidget(cnn_controls)
        
        cnn_group.setLayout(cnn_layout)
        layout.addWidget(cnn_group, 0, 1)
        
        # RNN section
        rnn_group = QGroupBox("Recurrent Neural Network")
        rnn_layout = QVBoxLayout()
        
        # RNN architecture controls
        rnn_controls = self.create_rnn_controls()
        rnn_layout.addWidget(rnn_controls)
        
        rnn_group.setLayout(rnn_layout)
        layout.addWidget(rnn_group, 1, 0)
        
        return widget
    
    def add_layer_dialog(self):
        """Open a dialog to add a neural network layer"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Neural Network Layer")
        layout = QVBoxLayout(dialog)
        
        # Layer type selection
        type_layout = QHBoxLayout()
        type_label = QLabel("Layer Type:")
        type_combo = QComboBox()
        type_combo.addItems(["Dense", "Conv2D", "MaxPooling2D", "Flatten", "Dropout"])
        type_layout.addWidget(type_label)
        type_layout.addWidget(type_combo)
        layout.addLayout(type_layout)
        
        # Parameters input
        params_group = QGroupBox("Layer Parameters")
        params_layout = QVBoxLayout()
        
        # Dynamic parameter inputs based on layer type
        self.layer_param_inputs = {}
        
        def update_params():
            # Clear existing parameter inputs
            for widget in list(self.layer_param_inputs.values()):
                params_layout.removeWidget(widget)
                widget.deleteLater()
            self.layer_param_inputs.clear()
            
            layer_type = type_combo.currentText()
            if layer_type == "Dense":
                units_label = QLabel("Units:")
                units_input = QSpinBox()
                units_input.setRange(1, 1000)
                units_input.setValue(32)
                self.layer_param_inputs["units"] = units_input
                
                activation_label = QLabel("Activation:")
                activation_combo = QComboBox()
                activation_combo.addItems(["relu", "sigmoid", "tanh", "softmax"])
                self.layer_param_inputs["activation"] = activation_combo
                
                params_layout.addWidget(units_label)
                params_layout.addWidget(units_input)
                params_layout.addWidget(activation_label)
                params_layout.addWidget(activation_combo)
            
            elif layer_type == "Conv2D":
                filters_label = QLabel("Filters:")
                filters_input = QSpinBox()
                filters_input.setRange(1, 1000)
                filters_input.setValue(32)
                self.layer_param_inputs["filters"] = filters_input
                
                kernel_label = QLabel("Kernel Size:")
                kernel_input = QLineEdit()
                kernel_input.setText("3, 3")
                self.layer_param_inputs["kernel_size"] = kernel_input
                
                params_layout.addWidget(filters_label)
                params_layout.addWidget(filters_input)
                params_layout.addWidget(kernel_label)
                params_layout.addWidget(kernel_input)
            
            elif layer_type == "Dropout":
                rate_label = QLabel("Dropout Rate:")
                rate_input = QDoubleSpinBox()
                rate_input.setRange(0.0, 1.0)
                rate_input.setValue(0.5)
                rate_input.setSingleStep(0.1)
                self.layer_param_inputs["rate"] = rate_input
                
                params_layout.addWidget(rate_label)
                params_layout.addWidget(rate_input)
        
        type_combo.currentIndexChanged.connect(update_params)
        update_params()  # Initial update
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add Layer")
        cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        def add_layer():
            layer_type = type_combo.currentText()
            
            # Collect parameters
            layer_params = {}
            for param_name, widget in self.layer_param_inputs.items():
                if isinstance(widget, QSpinBox):
                    layer_params[param_name] = widget.value()
                elif isinstance(widget, QDoubleSpinBox):
                    layer_params[param_name] = widget.value()
                elif isinstance(widget, QComboBox):
                    layer_params[param_name] = widget.currentText()
                elif isinstance(widget, QLineEdit):
                    # Handle kernel size or other tuple-like inputs
                    if param_name == "kernel_size":
                        layer_params[param_name] = tuple(map(int, widget.text().split(',')))
            
            self.layer_config.append({
                "type": layer_type,
                "params": layer_params
            })
            
            dialog.accept()
        
        add_btn.clicked.connect(add_layer)
        cancel_btn.clicked.connect(dialog.reject)
        
        dialog.exec()
    
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
        
        group.setLayout(layout)
        return group
    
    def create_cnn_controls(self):
        """Create controls for Convolutional Neural Network"""
        group = QGroupBox("CNN Architecture")
        layout = QVBoxLayout()
        
        # Placeholder for CNN-specific controls
        label = QLabel("CNN Controls (To be implemented)")
        layout.addWidget(label)
        
        group.setLayout(layout)
        return group
    
    def create_rnn_controls(self):
        """Create controls for Recurrent Neural Network"""
        group = QGroupBox("RNN Architecture")
        layout = QVBoxLayout()
        
        # Placeholder for RNN-specific controls
        label = QLabel("RNN Controls (To be implemented)")
        layout.addWidget(label)
        
        group.setLayout(layout)
        return group
    
    def train_neural_network(self):
        """Train the neural network with current configuration"""
        if not self.layer_config:
            self.show_error("Please add at least one layer to the network")
            return
        
        try:
            # Create and compile model
            model = self.create_neural_network()
            
            # Get training parameters
            batch_size = self.batch_size_spin.value()
            epochs = self.epochs_spin.value()
            learning_rate = self.lr_spin.value()
            
            # Prepare data for neural network
            if len(self.X_train.shape) == 1:
                X_train = self.X_train.reshape(-1, 1)
                X_test = self.X_test.reshape(-1, 1)
            else:
                X_train = self.X_train
                X_test = self.X_test
            
            # One-hot encode target for classification
            y_train = tf.keras.utils.to_categorical(self.y_train)
            y_test = tf.keras.utils.to_categorical(self.y_test)
            
            # Compile model
            optimizer = optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
            
            # Train model
            history = model.fit(X_train, y_train,
                                batch_size=batch_size,
                                epochs=epochs,
                                validation_data=(X_test, y_test),
                                callbacks=[self.create_progress_callback()])
            
            # Update visualization with training history
            self.plot_training_history(history)
            
            self.status_bar.showMessage("Neural Network Training Complete")
            
        except Exception as e:
            self.show_error(f"Error training neural network: {str(e)}")
    
    def create_neural_network(self):
        """Create neural network based on current configuration"""
        model = models.Sequential()
        
        # Add layers based on configuration
        for layer_config in self.layer_config:
            layer_type = layer_config["type"]
            params = layer_config["params"]
            
            if layer_type == "Dense":
                model.add(layers.Dense(**params))
            elif layer_type == "Conv2D":
                # Add input shape for the first layer
                if len(model.layers) == 0:
                    params['input_shape'] = self.X_train.shape[1:]
                model.add(layers.Conv2D(**params))
            elif layer_type == "MaxPooling2D":
                model.add(layers.MaxPooling2D())
            elif layer_type == "Flatten":
                model.add(layers.Flatten())
            elif layer_type == "Dropout":
                model.add(layers.Dropout(**params))
        
        # Add output layer based on number of classes
        num_classes = len(np.unique(self.y_train))
        model.add(layers.Dense(num_classes, activation='softmax'))
                
        return model
        
    def train_neural_network(self):
        """Train the neural network"""
        try:
            # Create and compile model
            model = self.create_neural_network()
            
            # Get training parameters
            batch_size = self.batch_size_spin.value()
            epochs = self.epochs_spin.value()
            learning_rate = self.lr_spin.value()
            
            # Compile model
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer,
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
            
            # Train model
            history = model.fit(self.X_train, self.y_train,
                              batch_size=batch_size,
                              epochs=epochs,
                              validation_data=(self.X_test, self.y_test),
                              callbacks=[self.create_progress_callback()])
            
            # Update visualization with training history
            self.plot_training_history(history)
            
        except Exception as e:
            self.show_error(f"Error training neural network: {str(e)}")
            
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
def main():
    """Main function to start the application"""
    app = QApplication(sys.argv)
    window = MLCourseGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
