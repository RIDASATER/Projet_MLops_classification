# **Projet MLOps : Classification de Texte**  
**Pipeline complet pour l'entraînement, le déploiement et le monitoring de modèles de NLP.**  

## **📌 Objectifs**  
- Mettre en place un pipeline MLOps **scalable et sécurisé** pour la classification de texte.  
- Versionner données/modèles avec **DVC** et suivre les expériences avec **MLflow**.  
- Déployer une API REST avec **FastAPI**, conteneurisée via **Docker/Kubernetes**.  
- Implémenter des bonnes pratiques de code (**générateurs, décorateurs, design patterns**).  
- Garantir la sécurité (**OAuth2, JWT, TLS**) et le monitoring (**Prometheus/Grafana**).  

---

## **⚙️ Fonctionnalités Clés**  

### **1. Gestion des Données & Versioning**  
- **DVC** pour le versioning des datasets et modèles.  
- Chargement optimisé via **générateurs/itérateurs** (économie de mémoire).  

### **2. Entraînement & Tracking**  
- Modèles implémentés : **BERT, LSTM, Regression Logistique**.  
- Suivi des métriques/hyperparamètres avec **MLflow**.  

### **3. Déploiement & API**  
- API REST avec **FastAPI**.  
- Conteneurisation via **Docker** et orchestration avec **Kubernetes**.  

### **4. Sécurité**  
- Authentification **OAuth2/JWT**.  
- Chiffrement des données (**AES/TLS**).  
- Contrôle d’accès **RBAC**.  

### **5. Monitoring**  
- Surveillance temps réel avec **Prometheus/Grafana**.  
- Détection automatique de **data drift** et alertes.  

### **6. Architecture Modulaire**  
- Design Patterns : **Singleton, Factory, Strategy**.  
- Décorateurs pour le logging/timing.  

---

## **🛠️ Stack Technique**  
- **Langage** : Python  
- **NLP** : Hugging Face Transformers, SpaCy  
- **ML** : PyTorch, TensorFlow, Scikit-learn  
- **MLOps** : DVC, MLflow, FastAPI, Docker, Kubernetes  
- **Monitoring** : Prometheus, Grafana  
- **CI/CD** : GitHub Actions/Jenkins  

---

## **🔒 Sécurité & Conformité**  
- Authentification **OAuth2** + **JWT**.  
- Chiffrement **TLS** (transfert) et **AES** (stockage).  
- Logs et audits des accès.  

---

## **🔄 Maintenance**  
- Réentraînement automatique en cas de drift.  
- Alertes sur les performances.  

---


## **🚀 Comment Démarrer**  
1. Cloner le dépôt :  
   ```bash
   git clone [URL]
   ```
2. Installer les dépendances :  
   ```bash
   pip install -r requirements.txt
   ```
3. Lancer l’API :  
   ```bash
   uvicorn src.api.main:app --reload
   ```
