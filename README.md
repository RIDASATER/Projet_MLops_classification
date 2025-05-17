# **MLOps Project: Text Classification**  
End-to-end pipeline for training, deploying, and monitoring NLP models.  

## **📌 Objectives**  
- Build a **scalable and secure** MLOps pipeline for text classification.  
- Version data/models with **DVC** and track experiments with **MLflow**.  
- Deploy a REST API using **FastAPI**, containerized via **Docker/Kubernetes**.  
- Implement coding best practices (**generators, decorators, design patterns**).  
- Ensure security (**OAuth2, JWT, TLS**) and monitoring (**Prometheus/Grafana**).  

---

## **⚙️ Key Features**  

### **1. Data Management & Versioning**  
- **DVC** for dataset and model versioning.  
- Optimized data loading using **generators/iterators** (memory-efficient).  

### **2. Training & Experiment Tracking**  
- Implemented models: **BERT, LSTM, Logistic Regression**.  
- Track metrics/hyperparameters with **MLflow**.  

### **3. Deployment & API**  
- REST API with **FastAPI**.  
- Containerization via **Docker** and orchestration with **Kubernetes**.  

### **4. Security**  
- **OAuth2/JWT** authentication.  
- Data encryption (**AES/TLS**).  
- **RBAC** access control.  

### **5. Monitoring**  
- Real-time monitoring with **Prometheus/Grafana**.  
- Automatic **data drift** detection and alerts.  

### **6. Modular Architecture**  
- Design Patterns: **Singleton, Factory, Strategy**.  
- Decorators for logging/timing.  

---

## **🛠️ Technical Stack**  
- **Language**: Python  
- **NLP**: Hugging Face Transformers, SpaCy  
- **ML**: PyTorch, TensorFlow, Scikit-learn  
- **MLOps**: DVC, MLflow, FastAPI, Docker, Kubernetes  
- **Monitoring**: Prometheus, Grafana  
- **CI/CD**: GitHub Actions/Jenkins  

---

## **🔒 Security & Compliance**  
- **OAuth2** + **JWT** authentication.  
- **TLS** (transfer) and **AES** (storage) encryption.  
- Access logs and auditing.  

---

## **🔄 Maintenance**  
- Automatic retraining on data drift.  
- Performance alerts.  

---

## **🚀 Getting Started**  
1. Clone the repository:  
   ```bash
   git clone [URL]
   ```
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the API:  
   ```bash
   uvicorn src.api.main:app --reload
   ```  
