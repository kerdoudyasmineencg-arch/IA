# 📘 TP INTELLIGENCE ARTIFICIELLE EN FINANCE  
## Analyse du risque et scoring crédit

**Nom & Prénom** : KERDOUD Yasmine et KHATER bassma
**Filière** : ENCG Settat – 4ᵉ année  
**Cours** : Intelligence Artificielle en Finance  
**Encadrant** : Pr. A. Larhlimi  
**Date** : 17/02/2026 

---

## 🔷 Introduction

Ce travail pratique s’inscrit dans le cadre du cours *Intelligence Artificielle en Finance* et vise à appliquer des méthodes statistiques et de machine learning à des problématiques concrètes de **gestion du risque financier** et de **scoring crédit**.

L’objectif principal est triple :  
- Analyser le risque de portefeuilles financiers à l’aide d’indicateurs statistiques (VaR, volatilité, Sharpe)  
- Utiliser le théorème de Bayes pour mettre à jour dynamiquement la probabilité de défaut d’un client  
- Construire et évaluer un modèle de classification supervisée (KNN) pour la détection des défauts de crédit  

L’approche adoptée est résolument **orientée métier**, en reliant systématiquement les résultats quantitatifs à des **décisions financières concrètes**.

---

## 🟦 PARTIE 1 — Statistiques et loi normale en finance

### 1.1 Statistiques descriptives des portefeuilles

Deux portefeuilles sont étudiés à partir de rendements mensuels observés sur 24 mois :

- **Portefeuille A (Conservatif)** : actions blue-chip européennes  
- **Portefeuille B (Agressif)** : actions small-cap technologiques  

Les indicateurs suivants ont été calculés :
- Moyenne et écart-type mensuels  
- Médiane  
- Rendement annualisé  
- Volatilité annualisée  

**Résultats synthétiques :**
- Le portefeuille A présente un rendement annuel modéré (~12 %) avec une volatilité faible (~1.6 %), indiquant une forte stabilité.
- Le portefeuille B affiche un rendement annualisé élevé (~41 %), mais accompagné d’une volatilité importante (~15 %), traduisant un niveau de risque élevé.

---

### 1.2 Visualisation des distributions

Les histogrammes et boxplots montrent :
- Une distribution concentrée et peu dispersée pour le portefeuille A
- Une distribution étalée avec de nombreux outliers pour le portefeuille B  

Ces graphiques confirment visuellement le **profil conservateur** du portefeuille A et le **caractère risqué** du portefeuille B.

---

### 1.3 Value at Risk (VaR 95 %)

La VaR paramétrique est calculée sous hypothèse de normalité :

\[
\text{VaR}_{95\%} = \mu - 1.65 \times \sigma
\]

**Interprétation :**
- La VaR annuelle du portefeuille A respecte la contrainte du client (perte maximale ≤ 50 000 €).
- Le portefeuille B dépasse largement cette contrainte, exposant le client à des pertes potentielles significatives.

Le test de normalité de Shapiro-Wilk indique que :
- Les rendements du portefeuille A sont compatibles avec la loi normale.
- Ceux du portefeuille B s’en écartent, rendant la VaR paramétrique moins fiable.

---

### 1.4 Ratio de Sharpe et recommandation

Le ratio de Sharpe permet d’évaluer le rendement ajusté du risque :

\[
\text{Sharpe} = \frac{R_{annuel} - r_f}{\sigma_{annuel}}
\]

Avec un taux sans risque de 3 % :
- Le portefeuille A présente un Sharpe stable et cohérent avec un profil défensif.
- Le portefeuille B, malgré un rendement élevé, est pénalisé par une volatilité excessive.

**Recommandation (Partie 1)** :  
👉 Pour un client avers au risque, le **portefeuille A** est recommandé.

---

## 🟩 PARTIE 2 — Théorème de Bayes et scoring crédit

### 2.1 Mise à jour du risque après retard de paiement

Un client du segment *Standard* (prior = 5 %) présente un retard de paiement.

En appliquant le théorème de Bayes :

\[
P(D|R) = \frac{P(R|D) \cdot P(D)}{P(R)}
\]

La probabilité de défaut passe de **5 % à ~30 %**, soit une multiplication du risque par **environ 6**.

**Décision métier** : Surveillance renforcée.

---

### 2.2 Mise à jour séquentielle

Deux semaines plus tard, un découvert bancaire > 500 € est observé.  
La probabilité de défaut dépasse alors **60 %**, indiquant un client à très haut risque.

Le graphique d’évolution montre clairement l’accumulation des signaux négatifs et leur impact sur le risque crédit.

---

### 2.3 Fonction générique Bayes

Une fonction Python générique a été développée afin de mettre à jour la probabilité de défaut pour tout événement observable.  
Cette fonction a été testée sur un client du segment *Risque*, confirmant l’explosion du risque après plusieurs événements défavorables.

---

### 2.4 Lien Bayes – Matrice de confusion

La précision issue de la matrice de confusion correspond exactement à :

\[
\text{Precision} = P(\text{Défaut} | \text{Retard})
\]

👉 Cela montre que la **précision d’un modèle de classification est une probabilité bayésienne a posteriori**.

---

## 🟥 PARTIE 3 — KNN et évaluation du modèle

### 3.1 Génération et exploration du dataset

Un dataset synthétique de 2000 clients a été généré (taux de défaut ≈ 16.7 %).  
Les variables les plus corrélées avec le défaut sont :
- Ratio dette / revenu  
- Historique des retards  

Les heatmaps et boxplots confirment leur pouvoir discriminant.

---

### 3.2 Prétraitement

- Séparation train/test (70/30) avec stratification  
- Normalisation via StandardScaler  

Cette étape est indispensable pour un algorithme basé sur les distances comme KNN.

---

### 3.3 Optimisation du paramètre K

Une validation croisée 5-fold a été réalisée pour plusieurs valeurs de K.

📌 **Résultat clé** :
- Le meilleur compromis est obtenu pour **K ≈ 25–30**
- AUC maximale ≈ **0.59**

Cela indique une capacité de discrimination modérée mais supérieure au hasard.

---

### 3.4 Évaluation du modèle final

Sur le jeu de test :
- AUC ≈ 0.59  
- Recall faible (défauts peu détectés)
- Specificity très élevée (> 99 %)

👉 Le modèle est **très conservateur**, évitant les faux positifs mais manquant de nombreux défauts.

---

### 3.5 Courbe ROC et seuil optimal

L’indice de Youden donne un seuil optimal ≈ **0.16**.

Des tests sur différents seuils montrent :
- Seuil bas → Recall ↑ mais Precision ↓
- Seuil élevé → Precision ↑ mais Recall ↓

---

## 🟨 Executive Summary — Partie 3 (Obligatoire)

- **Modèle choisi** : KNN avec K ≈ 25  
- **AUC** : ~0.59 (performance modérée)  
- **Recall / Precision** : compromis défavorable à la détection des défauts  
- **ROI** : meilleur pour un seuil bas, mais au prix d’analyses coûteuses  

👉 **Recommandation business** :
- Utiliser le modèle comme **outil de pré-filtrage**
- Coupler avec une analyse humaine pour les cas ambigus
- Envisager un modèle plus performant (Logistic Regression, Random Forest)

---

## 🟪 Conclusion

Ce TP a permis de relier des concepts théoriques (statistiques, Bayes, KNN) à des problématiques concrètes de finance et de gestion du risque.

Les principales difficultés rencontrées concernent :
- L’optimisation du compromis Recall / Precision
- L’interprétation métier des métriques ML

Ces méthodes sont directement applicables dans les domaines du **scoring crédit**, de la **gestion de portefeuille** et du **contrôle des risques**.

---

## 📚 Références

- Cours *Intelligence Artificielle en Finance*  
- Documentation Scikit-learn  
- Documentation NumPy / Pandas  
- Concepts de Value at Risk et Ratio de Sharpe
