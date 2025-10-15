# 🧠 Evaluating Classifier Robustness Against Adversarial Attacks

This project explores how well machine learning classifiers can withstand **adversarial attacks** — small, intentionally designed perturbations to input data that can drastically alter model predictions.  
The goal is to **analyze, compare, and improve model robustness** against such attacks.

---

## 📘 Overview
Machine learning models often perform well on clean data but can fail when exposed to adversarial examples.  
This project evaluates different classifiers under adversarial scenarios and implements defense strategies to improve their resilience.

---

## ⚙️ Features
- Implementation of **common adversarial attack techniques** (FGSM, PGD, etc.)  
- Evaluation of **multiple classifiers** (Logistic Regression, SVM, CNN, etc.)  
- Visualization of adversarial effects on model predictions  
- Performance comparison between **clean vs. adversarial data**  
- Defensive training and regularization methods to enhance robustness  

---

## 🧰 Tech Stack
- **Language:** Python  
- **Libraries:** TensorFlow / PyTorch, NumPy, Matplotlib, scikit-learn  
- **Environment:** Jupyter Notebook / Google Colab  

---

## 🧪 Methodology
1. **Data Preparation:** Import and preprocess dataset (e.g., MNIST, CIFAR-10).  
2. **Model Training:** Train baseline models on clean data.  
3. **Adversarial Attack Simulation:** Generate adversarial samples using FGSM and PGD.  
4. **Evaluation:** Measure model accuracy and confidence drop.  
5. **Defense Implementation:** Apply adversarial training and regularization techniques.  

---

## 📊 Results
- Models show a significant accuracy drop under adversarial attacks.  
- Adversarial training helps recover performance partially.  
- CNN models show higher robustness than linear models when properly regularized.  

---

## 💡 Key Learnings
- Adversarial robustness is crucial for ML models in security-sensitive applications.  
- Small perturbations can easily fool high-accuracy models.  
- Defensive training and regularization significantly improve reliability.

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/Mmeetttt/adversarial-attack-robustness.git
   cd adversarial-attack-robustness
2. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Run the Jupyter notebook or Python script:
```bash
jupyter notebook
# or
python main.py
