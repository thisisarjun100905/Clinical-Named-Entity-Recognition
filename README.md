# Clinical-Named-Entity-Recognition
Here’s a well-structured and visually appealing README file for your project:

---
This repository implements **Clinical Named Entity Recognition** using **ClinicalBERT** for feature extraction and a **Dense Neural Network** for classification. The project focuses on analyzing discharge summaries, identifying medical entities, and categorizing them into types like "problem," "procedure," and "anatomical structure."

---

## 📖 **Overview**

- Extract and classify clinical entities from discharge summaries.
- Utilize **ClinicalBERT** for advanced tokenization and feature extraction.
- Train a **Dense Neural Network** to classify entities into pre-defined categories.
- Achieve a classification accuracy of ~90%.

---

## 📊 **Dataset**

The dataset includes **discharge summaries** annotated with medical entities. Each entity is characterized by:
- **Text**: Entity string (e.g., "cancer").
- **Category**: The type of the entity (e.g., "problem").
- **Confidence score**: Probability of correct classification.
- **Position**: Character range of the entity within the summary.

### **Example Input Data**
| **Discharge Summary** | **Entity** |
|------------------------|------------|
| Patient shows signs of cancer with associated pain. | {'[[cancer, problem, 0.934, 24-30], [pain, symptom, 0.876, 45-49]]'} |
| Thoracotomy syndrome observed in post-surgery patients. | {'[[Thoracotomy syndrome, problem, 0.95, 0-20]]'} |

---

## 🛠️ **Data Processing**

### **1. Parsing Entities**
Entities in string format are parsed into structured data for better analysis. 

#### Code Snippet:
```python
import ast

def parse_entity(entity_str):
    try:
        entity_list = ast.literal_eval(entity_str)
        parsed_data = []
        for item in entity_list:
            if len(item) == 4:
                parsed_data.append({
                    "text": item[0],
                    "category": item[1],
                    "confidence": item[2],
                    "position": item[3]
                })
        return parsed_data
    except Exception as e:
        print(f"Error parsing entity: {e}")
        return []
```

### **2. Flattening Data**
Transform parsed entities into a structured DataFrame for processing.

| **Discharge Summary** | **Entity Text**        | **Category** | **Confidence** | **Position** |
|------------------------|------------------------|--------------|----------------|--------------|
| Patient shows signs of cancer with associated pain. | cancer                 | problem      | 0.934          | 24-30       |
| Patient shows signs of cancer with associated pain. | pain                   | symptom      | 0.876          | 45-49       |
| Thoracotomy syndrome observed in post-surgery patients. | Thoracotomy syndrome  | problem      | 0.95           | 0-20        |

---

## 🔍 **Feature Extraction**

- **Tokenization**: Performed using ClinicalBERT.  
- **Embeddings**: Generate embeddings from ClinicalBERT's final hidden layer, capturing the semantic meaning of the entities.

---

## 🧑‍💻 **Model Training**

### **Model Architecture**
- Input layer: Accepts ClinicalBERT embeddings.
- Hidden layers: Two dense layers with ReLU activation.
- Output layer: Softmax activation for multi-class classification.

### **Training Details**
- Optimizer: Adam
- Loss: Sparse Categorical Cross-Entropy
- Metrics: Accuracy
- Train-Test Split: 80%-20%

#### Code Snippet:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(np.unique(y)), activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
```

---

## 🔮 **Predictions**

- **Output**: Encoded labels decoded into human-readable categories using a mapping dictionary.

#### Example Code for Decoding:
```python
predicted_labels = [key for pred in y_pred for key, value in label_dict.items() if value == pred]
print(predicted_labels)
```

---

## 📈 **Results**

The model achieves a **~90% classification accuracy** on the test dataset. Additional evaluation metrics:
- **Precision**
- **Recall**
- **F1-Score**

---

## 📊 **Workflow Diagram**

Below is the high-level workflow for the project:

```
Discharge Summary → Tokenization → Feature Extraction (ClinicalBERT) → 
Entity Parsing → Encoding → Neural Network → Classification
```

---

## 🚀 **Future Work**

1. Fine-tune ClinicalBERT on domain-specific datasets for enhanced performance.
2. Implement advanced tokenization techniques for complex medical terminology.
3. Add support for additional entity categories and subcategories.
4. Explore semi-supervised learning for limited annotation datasets.

---

## 🧑‍💻 **Dependencies**

- Python 3.8+
- pandas
- numpy
- TensorFlow/Keras
- scikit-learn
- transformers

---

## 📜 **Author**

This project was implemented by **Arjun**.

--- 
