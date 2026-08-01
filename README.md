📰 AI-Powered Fake News Detection Using Text Classification
A from-scratch machine learning pipeline that classifies a news article as Real or Fake using only its textual content — no external fact-checking APIs, no publisher metadata, no knowledge-graph lookups. Just the language itself.
Built during the AI & ML Summer Internship Program at the Indian Institute of Computing and Technology (IICT), this project takes the full supervised-learning journey: raw data → cleaning → TF-IDF vectorization → four competing classifiers → evaluation → a live, deployed Streamlit app.
�
￼ 

�
￼ ￼ ￼ ￼ ￼ 

📌 Table of Contents
Overview
Why Fake News Detection Matters
Dataset
Methodology
1. Text Preprocessing
2. Feature Extraction (TF-IDF)
3. Train-Test Split
4. Models Trained
Results
Parametric vs. Non-Parametric — Key Insight
System Architecture & Deployment
Tech Stack
Project Structure
Getting Started
What I Learned
Limitations
Future Scope
References
Author
🔍 Overview
Problem Statement: Build a supervised ML pipeline that labels a news article as real or fake using only the statistical properties of the text itself — no author history, no publisher reputation, no social engagement signals.
This constraint is deliberate. It isolates how much discriminative signal language alone carries, which is the hardest and most generalizable version of the problem. The pipeline covers:
Data acquisition & cleaning
Text preprocessing & normalization
Feature extraction (TF-IDF)
Training 4 classifiers across parametric and non-parametric model families
Rigorous multi-metric evaluation (not just accuracy)
Packaging the winning model into a real-time, browser-based inference app
⚠️ Why Fake News Detection Matters
Asymmetric cost of production vs. verification — fabricating an article takes minutes; manual fact-checking takes hours or days.
Virality outpaces correction — false stories consistently spread faster and reach wider audiences than the corrections that follow them.
Scale — manual fact-checking cannot keep up with the volume published daily across news aggregators, blogs, and social platforms.
Erosion of institutional trust — repeated exposure to fabricated content degrades trust in legitimate journalism, an effect that's slow to reverse.
Downstream risk — in public health, finance, and elections, a single viral fabrication can cause real damage before any correction lands.
These factors make the case for lightweight, automated, first-pass classifiers that can run at the point of publication or consumption — not just centralized human fact-checking.
📊 Dataset
This project uses WELFake (Word Embedding over Linguistic Features for Fake News Detection) — a corpus built by merging four widely-used news datasets (Kaggle, McIntire, Reuters, BuzzFeed Political) specifically to reduce single-source overfitting.
�
￼ 

Detail
Value
Source
WELFake (Kaggle) — merged from Kaggle, McIntire, Reuters & BuzzFeed Political
Total articles (post-cleaning)
72,095
Fake articles (label = 0)
35,028 (48.6%)
Real articles (label = 1)
37,067 (51.4%)
Fields used
title, text, label
Train / Test split
80% / 20%, stratified
Test set size
14,419 articles
Each record was cleaned by combining title + text, dropping nulls, and shuffling with a fixed seed for reproducibility.
🧪 Methodology
1. Text Preprocessing
Raw article text goes through a deliberate cleaning order:
Lowercasing
Removal of source-agency bylines (e.g. (City – Reuters) and the literal token reuters) — this prevents the model from learning a trivial source-identity shortcut instead of a genuine content-based signal
URL removal
Removal of all non-alphabetic characters
Stop-word removal after whitespace tokenization
Discarding single-character tokens as noise
�
Example: raw vs. cleaned text
Raw:
Trump's ongoing meltdown over fake news (the rest of us call it reporting) organizations entered what seems like its eighteenth year on Wednesday after NBC correctly reported...
Cleaned:
trump ongoing meltdown fake news rest us call reporting organizations entered seems like eighteenth year wednesday nbc correctly reported...
�

2. Feature Extraction (TF-IDF)
Cleaned text is vectorized using TF-IDF, chosen over raw bag-of-words counts because it downweights high-frequency, low-information terms while preserving discriminative rare terms.
Top 3,000–5,000 highest-scoring terms retained
min_df = 3 — discards overly rare / noisy terms
max_df = 0.9 — discards corpus-specific "stop-words" that carry little discriminative value
3. Train-Test Split
Stratified 80/20 split (random_state = 42) to preserve class balance across both partitions and keep results reproducible.
4. Models Trained
Four classifiers spanning parametric and non-parametric families were trained under identical feature and split conditions:
Model
Family
Core Idea
Logistic Regression
Parametric
Learns a fixed-size weight vector over TF-IDF terms; linear log-odds decision boundary
Random Forest
Ensemble / Non-parametric
100 bootstrapped decision trees, majority-vote aggregated; complexity grows with the data
Neural Network (MLP)
Parametric
Single hidden layer (64–100 neurons), trained via back-propagation with early stopping
K-Nearest Neighbours
Non-parametric
Majority vote among 5–7 nearest neighbours in TF-IDF space; no explicit training phase
🏆 Results
�
￼ 

Model
Accuracy
Precision
Recall
F1-Score
🥇 Random Forest (Ensemble)
96.01%
0.9571
0.9656
0.9613
🥈 Neural Network (MLP)
95.90%
0.9559
0.9648
0.9605
🥉 Logistic Regression
95.46%
0.9524
0.9597
0.9556
K-Nearest Neighbours
68.31%
0.6242
0.9639
0.7580
(Metrics computed on the 14,419-article held-out test set.)
�
￼ 

Random Forest came out on top and was the model serialized for deployment — competitive accuracy, strong recall on the "real" class, and it doesn't rely on raw distance in a 3,000+ dimensional sparse space (unlike KNN).
Most Discriminative Terms
Random Forest's Gini importance surfaced the terms the model actually leans on:
�
￼ 

Notably, wire-service and formatting artifacts (said, via, image, washington) dominate — a reminder that lexical classifiers partly key off stylistic fingerprints of how an article was written/sourced, not just semantic truthfulness.
🧠 Parametric vs. Non-Parametric — Key Insight
Parametric models (Logistic Regression, Neural Network) learn a fixed-size parameter set that doesn't grow with the training data — they assume a specific functional form (linear, or layered non-linear).
Non-parametric models (Random Forest, KNN) let complexity scale with the data itself — Random Forest via the number/depth of trees, KNN via the entire training set it retains as its "model."
What actually happened:
Logistic Regression trained almost instantly and stayed fully interpretable via per-term coefficients.
The Neural Network took longer to train but captured mild non-linear interactions.
Random Forest was the slowest to train (~24s vs. ≤5s for the others) but offered a complementary interpretability lens via feature importances, and edged out the highest accuracy.
KNN broke down — in 3,000+ TF-IDF dimensions, pairwise Euclidean distances stop being discriminative (the classic curse of dimensionality), so the neighborhood structure KNN depends on collapses. This isn't evidence that non-parametric methods are bad for text; it's evidence that distance-based methods need dimensionality reduction or dense embeddings to work well on raw sparse bag-of-words features.
Takeaway: for sparse, high-dimensional lexical features like TF-IDF, models that learn a global weighting over features — whether parametric or tree-ensemble — generalize noticeably better than an instance-based distance method operating directly in the raw feature space.
🏗️ System Architecture & Deployment
The best-performing model + the fitted TF-IDF vectorizer were serialized (pickle) and wired into a Streamlit web app, built and tested locally in VS Code. Users paste an article's text into a browser input box and get an instant Fake/Real prediction — no need to touch the training code.
�
￼ 

Flow: User input → Streamlit UI → Preprocessing module (clean + tokenize) → pre-fitted TF-IDF vectorizer → trained model bank → majority-vote ensemble → prediction + confidence → rendered back on the UI.
🛠️ Tech Stack
Layer
Tools
Language
Python
Data handling
pandas, NumPy
NLP / Feature extraction
TF-IDF (scikit-learn), custom regex-based cleaning
Modeling
scikit-learn (LogisticRegression, RandomForestClassifier, MLPClassifier, KNeighborsClassifier)
Visualization
Matplotlib, Seaborn
Deployment
Streamlit
Development
Google Colab (training), VS Code (app development)
Serialization
Pickle
📂 Project Structure
Adjust this to match your actual repo layout — a suggested structure based on the pipeline is below.
fake-news-detection/
├── assets/                      # Charts & diagrams used in this README
│   ├── class_distribution.png
│   ├── accuracy_comparison.png
│   ├── metric_comparison.png
│   ├── confusion_matrix_grid.png
│   ├── system_architecture.png
│   └── top_features.png
├── notebooks/
│   └── model_training.ipynb     # Full training/eval pipeline (Colab)
├── app/
│   └── app.py                   # Streamlit inference app
├── models/
│   ├── models.pkl                # Serialized trained classifiers
│   └── vectorizer.pkl            # Fitted TF-IDF vectorizer
├── requirements.txt
└── README.md
🚀 Getting Started
Prerequisites
Python 3.10+
pip
Installation
# Clone the repository
git clone https://github.com/crypticcoders/fake-news-detection.git
cd fake-news-detection

# Install dependencies
pip install -r requirements.txt
Run the app locally
streamlit run app/app.py
Then open the local URL Streamlit prints (typically http://localhost:8501) and paste in any article text to get a live prediction.
Retrain the models
Open notebooks/model_training.ipynb in Google Colab (or Jupyter), mount your dataset, and run all cells — it reproduces preprocessing, TF-IDF vectorization, training, evaluation, and the serialized models.pkl / vectorizer.pkl artifacts used by the app.
💡 What I Learned
How to build a complete supervised NLP pipeline from first principles — no pre-built fake-news classifier, no external fact-checking API.
Why removing source-identity tokens (like reuters) before vectorizing matters — otherwise the model just memorizes "this dataset's Reuters articles" instead of learning content-based signal.
TF-IDF vs. raw counts — and why min_df/max_df pruning meaningfully changes what the vectorizer treats as signal vs. noise.
The practical difference between parametric and non-parametric models isn't just academic — it shows up directly in training time, interpretability, and how each model fails.
Why KNN struggles in high-dimensional sparse spaces (curse of dimensionality) — and that this is a property of the representation, not a verdict on non-parametric methods in general.
How to evaluate a classifier properly: accuracy alone hides class-wise failure modes — precision/recall/F1/confusion matrices matter because false positives and false negatives carry different real-world costs here.
End-to-end model serialization and deployment — going from a trained scikit-learn object in a notebook to a live, interactive Streamlit app that a non-technical user can actually use.
⚠️ Limitations
TF-IDF captures lexical frequency only — it ignores word order, syntax, and deeper semantics.
Due to compute constraints, some experiments were run on a stratified subsample rather than the full 72,095-article corpus.
Even after removing the literal token reuters, other stylistic/formatting fingerprints tied to a specific source dataset may still let a model exploit spurious correlations rather than purely content-based truthfulness signals — a known risk with datasets merged from multiple sources.
🔮 Future Scope
Retrain on the full corpus with k-fold cross-validation and systematic hyperparameter tuning.
Explore dense embeddings (Word2Vec, GloVe) or transformer encoders (e.g., BERT) to capture semantic/contextual signal beyond raw term frequency.
Add explainability (SHAP/LIME) to the app so users can see which words drove a given prediction.
Expose the model via a REST API and incorporate source-credibility metadata as an auxiliary feature.
📚 References
P. K. Verma, P. Agrawal, I. Amorim, R. Prodan, "WELFake: Word Embedding Over Linguistic Features for Fake News Detection," IEEE Transactions on Computational Social Systems, 2021.
F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," JMLR, vol. 12, 2011.
L. Breiman, "Random Forests," Machine Learning, vol. 45, 2001.
Streamlit Inc., Streamlit Documentation
👤 Author
Christy Joyce A First-year ECE student, VIT Chennai AI & ML Summer Intern, Indian Institute of Computing and Technology (IICT)
GitHub: @crypticcoders
�
⭐ If this project was useful or interesting, consider giving it a star!

