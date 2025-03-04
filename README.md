# Airline Delay Prediction 

##  Project Overview

This project aims to predict **weather-related flight delays** using a deep neural network (DNN). Given historical airline delay data, the model classifies whether a flight will be **delayed by more than 100 minutes** due to weather conditions.

##  Dataset

The dataset used is `Airline_Delay_Cause.csv`, containing flight delay information, including:

- Flight number, carrier, and airport details
- Weather-related delays in minutes
- Other delay causes (security, carrier, etc.)

### **Target Variable**

A new binary column `` was created:

- `1` → Weather delay **> 100 minutes**
- `0` → Weather delay **≤ 100 minutes**

##  Installation & Setup

### **1. Clone the Repository**

```sh
git clone https://github.com/your-username/airline-delay-prediction.git
cd airline-delay-prediction
```

### **2. Install Dependencies**

```sh
pip install -r requirements.txt
```

### **3. Run the Jupyter Notebook**

```sh
jupyter notebook DNN.ipynb
```

##  Model Architecture

The deep learning model is a **fully connected neural network** built using Keras:

```python
KerasModel = keras.models.Sequential([
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])
```

- **ReLU activations** to prevent vanishing gradients.
- **Dropout (0.2)** for regularization.
- **Sigmoid activation** in the output layer (for binary classification).

##  Evaluation & Results

The model is trained and evaluated using **binary cross-entropy loss** and **AdamW optimizer**.

```python
KerasModel.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

**Expected Accuracy:** \96%



##  License

This project is licensed under the MIT License. See `LICENSE` for details.

##  Acknowledgments

- TensorFlow & Keras for model development
- Open-source flight delay datasets

---

Made with  by [Mohamed Tarek]

