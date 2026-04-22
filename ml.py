# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 03:49:06 2025

@author: HP
"""

#%% biblioteke
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA



from scipy.stats import spearmanr

#%%

def run_grid_search(model, param_grid, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1, verbose=0):
    """
    Performs GridSearchCV and returns the best estimator and parameters.
    """
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose
    )
    grid_search.fit(X_train, y_train)
    
    print("Best parameters:", grid_search.best_params_)
    print("Best CV score:", grid_search.best_score_)
    
    return grid_search.best_estimator_, grid_search.best_params_


def evaluate_model(model, X, y, positive_class=1):
    """
    Evaluates a model and prints common metrics.
    Returns metrics in a dictionary.
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, positive_class] if hasattr(model, "predict_proba") else None
    
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp)
    auc = roc_auc_score(y, y_proba) if y_proba is not None else None
    
    print("Confusion Matrix:\n", cm)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall (Sensitivity): {rec:.4f}")
    print(f"Specificity: {spec:.4f}")
    if auc is not None:
        print(f"ROC AUC: {auc:.4f}")
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "roc_auc": auc,
        "confusion_matrix": cm
    }


#%% UCITAVANJE PODATAKA

df = pd.read_csv('.//career_change_prediction_dataset.csv')


print(df.head)


#broj redova
print(len(df))
print(df.shape[0])


#%%PROVERA NULL VRENODSTI

print(df.isnull().sum())
#udeo null vrednosti
print(df["Family Influence"].isnull().sum() / len(df["Family Influence"]))

#brisanje redova sa null vrednostima
df.dropna(subset=["Family Influence"],inplace=True)
print(df.isnull().sum())


#%% PROVERA OUTLIER-A - NEMA IH 

num_cols = ["Age", "Years of Experience", "Salary", "Job Opportunities"]
for col in num_cols:
    plt.figure(figsize=(6,4))
    sn.boxplot(x=df[col], showfliers=True)
    plt.title(f"Boxplot - {col}")
    plt.show()

# provera tipova podataka:
print(df.dtypes)

print(df.describe())
print(df.describe(include=["object", "category"]))


# %%ANALIZA KATEGORIČKIH PODATAKA


def plot_pie(df, column="Field of Study"):
    """
    Crta pie chart raspodele kategorija u koloni
    """
    counts = df[column].value_counts()
    plt.figure(figsize=(8,8))
    plt.pie(
        counts,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor":"k"}
    )
    plt.title(f"Distribucija - {column}")
    plt.show()


# field of study
plot_pie(df, "Field of Study")

# education level
plot_pie(df, "Education Level")


# worklife balance
def categorize(x):
    if x in [1, 2, 3]:
        return "Low"
    elif x in [4, 5, 6, 7]:
        return "Medium"
    elif x in [8, 9, 10]:
        return "High"
    else:
        return None

df_cat = df.copy()
df_cat["Work-Life Balance"] = df_cat["Work-Life Balance"].apply(categorize)
plot_pie(df_cat, "Work-Life Balance")

#Current Occupation

print('Current Occupation:')
print(df['Current Occupation'].unique())
print('Number of unique values: ', df['Current Occupation'].nunique())

broj = df['Current Occupation'].value_counts()

print("Most common occupation:",broj.idxmax(), "with", broj.max(), "workers")

print(broj)

sn.barplot(x=broj.values,y=broj.index)
plt.xlabel("Count")
plt.ylabel("Occupation")
plt.title("Number of People per Occupation")
plt.show()

#Job Satisfaction

bins = [0,3,7,10]
labels = ['Low(1-3)','Medium(4-7)','High(8-10)']


df_satisfaction = pd.cut(df['Job Satisfaction'], bins = bins,labels=labels,include_lowest=True)

num_satisf = df_satisfaction.value_counts()

sn.barplot(x = num_satisf.index, y = num_satisf.values)
plt.xlabel("Satisfaction")
plt.ylabel("Count")
plt.title("Job Satisfaction")
plt.show()


#Industry Growth Rate

num_growth = df['Industry Growth Rate'].value_counts()

sn.barplot(x=num_growth.index, y = num_growth.values)
plt.xlabel("Industry Growth Rate")
plt.ylabel("Count")
plt.show()


#KOJE PODATKE KORISTIMO

#da li enkodirati Industry Growth Rate - SAMA KOLONA NIJE KORISNA

#Industry Growth Rate

mapping_industry = {"Low": 1, "Medium": 2, "High": 3}
df_num = df['Industry Growth Rate'].map(mapping_industry)
 
corr, p_value = spearmanr(df_num, df["Likely to Change Occupation"])

print("Spearman correlation:", corr)
print("p-value:", p_value)
 
# p vrednost blizu 0, nema neki uticaj

#Job Satisfaction - KORISNO

sn.barplot(x="Job Satisfaction", y="Likely to Change Occupation", data=df, estimator=lambda x: sum(x)/len(x))

plt.title("Proportion of target=1 by Job Satisfaction")
plt.show()

corr, p = spearmanr(df["Job Satisfaction"], df["Likely to Change Occupation"])
print("Spearman correlation:", corr, "p-value:", p)

#gledamo koji broj ljudi menja karijeru na osnovu toga koliko su zadovoljni sa poslom

change_str = df['Likely to Change Occupation'].map({1:'YES',0:'NO'})

plt.figure(figsize=(10,6))
sn.countplot(data=df, x='Job Satisfaction', hue=change_str)
plt.xlabel('Job Satisfaction')
plt.ylabel('Number of People')
plt.title('Occupation Change vs Job Satisfaction')
plt.show()


#utice na verovatnocu promene karijere



#Current Occupation ne utice toliko na promenu karijere

plt.figure(figsize=(10,6))
sn.countplot(data=df, y='Current Occupation', hue=change_str)
plt.xlabel('Current Occupation')
plt.ylabel('Number of People')
plt.title('Occupation Change vs Job Satisfaction')
plt.show()


#%% ENCODING


# education level
education_map = {
    "High School": 0,
    "Bachelor's": 1,
    "Master's": 2,
    "PhD": 3
}
df["Education Level Encoded"] = df["Education Level"].map(education_map)


# Gender
gender_map = {
    "Male" : 0,
    "Female" : 1
}
df["Gender"] = df["Gender"].map(gender_map)



#Family Influence
family_map = {
    'None':0,
    'Low':1,
    'Medium':2,
    'High':3
    }

df['Family Influence Encoded'] = df['Family Influence'].map(family_map)

# Field of study - da li uopste treba prebaciti u numericki podatak i pravi li neku razliku u analizi
print(df.groupby("Field of Study")["Likely to Change Occupation"].mean().sort_values())
# Dobijeni rezultati su u intervalu od 0,55 do 0,59 sto i nije neki veliki uticaj, zbog toga necemo 
# enkodirati, vec ostaviti u trenutnom obliku.


#%% STANDARDIZACIJA


# Drop 'Salary' and 'Job Opportunities'


numeric_features = [
    'Age', 'Years of Experience', 'Job Satisfaction', 'Work-Life Balance',
    'Job Opportunities', 'Salary', 'Job Security', 'Skills Gap',
    'Professional Networks', 'Career Change Events', 'Technology Adoption'
]

binary_features = [
    'Career Change Interest', 'Mentorship Available', 'Certifications',
    'Freelancing Experience', 'Geographic Mobility','Gender'
] #gender dodat vamo zato sto je enkodiran u 0 i 1


#encodovani su vec
# categorical_features = [
#      'Education Level',
#      'Family Influence'
# ]

#features_to_scale = numeric_features + categorical_features
features_to_scale = numeric_features + ['Education Level Encoded', 'Family Influence Encoded']


scaler = StandardScaler()

df_scaled = df.copy()

df_scaled = df_scaled.drop(['Current Occupation', 'Industry Growth Rate','Field of Study'], axis=1)



df_scaled[features_to_scale] = scaler.fit_transform(df[features_to_scale])


print(df_scaled.head())

#%%
X = df_scaled.drop("Likely to Change Occupation", axis=1)
y = df_scaled["Likely to Change Occupation"]



X = df_scaled.drop(["Likely to Change Occupation", 
                    "Education Level", 
                    "Family Influence"], axis=1)

#%%------------------------------------------kNN----------------------------------------------------
# sa samo najbitnijim obelezjima koje smo videli preko random foresta

#deljenje na skupove za trening, validaciju i test

# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# rf.fit(X_train, y_train)

# # Get importances
# importances = pd.Series(rf.feature_importances_, index=X_train.columns)
# importances = importances.sort_values(ascending=False)

# print("Feature importances:\n", importances)


# plt.figure(figsize=(8,5))
# sns.barplot(x=importances.values, y=importances.index)
# plt.title("Feature Importances")
# plt.show()


print("*********************************************************************\n")
print("*********kNN mali skup*****************\n")
print("*********************************************************************")

Xknn = X[['Career Change Interest','Job Satisfaction','Salary','Age']]


param_grid_knn = {
    'n_neighbors': list(range(1, 25, 2)),  # odd numbers 1-23
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

#%%


print("Columns in X:", Xknn.columns)


X_temp, X_test, y_temp, y_test = train_test_split(
    Xknn, y, test_size=0.2, random_state=42, stratify=y
)


# 87.5% of 0.8 = 0.7 train, 0.1 validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp
)


print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)


#%%

knn = KNeighborsClassifier()

best_knn, best_params_knn = run_grid_search(knn, param_grid_knn, X_train, y_train)


print("VALIDACIJA:")
val_metrics = evaluate_model(best_knn, X_val, y_val)

print("TEST:")
test_metrics = evaluate_model(best_knn, X_test, y_test)



#%% -------------------------------------------kNN sa vise featura-----------------------------------------------------------------

#da bi pokazali sa PCA mi moramo da dodamo vise feature-a, zato sto su nam oni gore svi potrebni da bi odrzali 95%
#varijanse kod koriscenja pca.


#deljenje na skupove za trening, validaciju i test
print("*********************************************************************\n")
print("**********kNN sa vise featura*****************\n")
print("*********************************************************************")




print("Columns in X:", X.columns)


X_temp1, X_test1, y_temp1, y_test1 = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 87.5% of 0.8 = 0.7 train, 0.1 validation
X_train1, X_val1, y_train1, y_val1 = train_test_split(
    X_temp1, y_temp1, test_size=0.125, random_state=42, stratify=y_temp1
)



print("Train:", X_train1.shape, y_train1.shape)
print("Validation:", X_val1.shape, y_val1.shape)
print("Test:", X_test1.shape, y_test1.shape)


#%%


knn1 = KNeighborsClassifier()

best_knn1, best_params_knn1 = run_grid_search(knn1, param_grid_knn, X_train1, y_train1)

print("VALIDACIJA:")
val_metrics = evaluate_model(best_knn1, X_val1, y_val1)

print("TEST:")
test_metrics = evaluate_model(best_knn1, X_test1, y_test1)




#%% -----------------------------------kNN sa PCA--------------------------------------------------------------



#deljenje na skupove za trening, validaciju i test
print("*********************************************************************\n")
print("*******************kNN sa PCA*************************\n")
print("*********************************************************************")


pca = PCA(n_components=0.95)  # 95% varijanse sacuvamo
X_pca = pca.fit_transform(X)
print("Original shape:", X.shape)
print("Reduced shape:", X_pca.shape)


print("Columns in X:", X.columns)



X_temp2, X_test2, y_temp2, y_test2 = train_test_split(
    X_pca, y, test_size=0.2, random_state=42, stratify=y
)


# 87.5% of 0.8 = 0.7 train, 0.1 validation
X_train2, X_val2, y_train2, y_val2 = train_test_split(
    X_temp2, y_temp2, test_size=0.125, random_state=42, stratify=y_temp2
)



print("Train:", X_train2.shape, y_train2.shape)
print("Validation:", X_val2.shape, y_val2.shape)
print("Test:", X_test2.shape, y_test2.shape)


#%%


knn2 = KNeighborsClassifier()


best_knn2, best_params_knn2 = run_grid_search(knn2, param_grid_knn, X_train2, y_train2)

print("-----------------------------VALIDACIJA:")
val_metrics = evaluate_model(best_knn2, X_val2, y_val2)

print("----------------------------------TEST:")
test_metrics = evaluate_model(best_knn2, X_test2, y_test2)




#%% -----------------------------------------------------SVM--------------------------------------------------------




#deljenje na skupove za trening, validaciju i test
print("*********************************************************************\n")
print("****************SVM mali skup*************************\n")
print("*********************************************************************")

param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']  # for rbf/poly kernels
}




X3= X[['Career Change Interest','Job Satisfaction','Salary','Age']]



print("Columns in X:", X3.columns)



X_temp3, X_test3, y_temp3, y_test3 = train_test_split(
    X3, y, test_size=0.2, random_state=42, stratify=y
)


# 87.5% of 0.8 = 0.7 train, 0.1 validation
X_train3, X_val3, y_train3, y_val3 = train_test_split(
    X_temp3, y_temp3, test_size=0.125, random_state=42, stratify=y_temp3
)



print("Train:", X_train3.shape, y_train3.shape)
print("Validation:", X_val3.shape, y_val3.shape)
print("Test:", X_test3.shape, y_test3.shape)

#%%



svm_model = SVC(probability=True, random_state=42)

best_svm, best_params_svm = run_grid_search(svm_model, param_grid_svm, X_train3, y_train3)


print("-----------------------------VALIDACIJA:")
val_metrics = evaluate_model(best_svm, X_val3, y_val3)

print("----------------------------------TEST:")
test_metrics = evaluate_model(best_svm, X_test3, y_test3)





#%%------------------------------------------ SVM sa vise feature-a---------------------------------------------------





#deljenje na skupove za trening, validaciju i test
print("*********************************************************************\n")
print("***********************  SVM sa vise feature-a*********************\n")
print("*********************************************************************")



print("Columns in X:", X.columns)



X_temp4, X_test4, y_temp4, y_test4 = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 87.5% of 0.8 = 0.7 train, 0.1 validation
X_train4, X_val4, y_train4, y_val4 = train_test_split(
    X_temp4, y_temp4, test_size=0.125, random_state=42, stratify=y_temp4
)



print("Train:", X_train4.shape, y_train4.shape)
print("Validation:", X_val4.shape, y_val4.shape)
print("Test:", X_test4.shape, y_test4.shape)

#%%


svm_model4 = SVC(probability=True, random_state=42)

best_svm4, best_params_svm = run_grid_search(svm_model4, param_grid_svm, X_train4, y_train4)

print("-----------------------------VALIDACIJA:")
val_metrics = evaluate_model(best_svm4, X_val4, y_val4)

print("----------------------------------TEST:")
test_metrics = evaluate_model(best_svm4, X_test4, y_test4)




#%% -----------------SVM sa PCA------------------------------------------------





#deljenje na skupove za trening, validaciju i test
print("*********************************************************************\n")
print("***********************  SVM sa PCA*********************\n")
print("*********************************************************************")





pca = PCA(n_components=0.95)  # keep 95% of variance
X_pca5 = pca.fit_transform(X)
print("Original shape:", X.shape)
print("Reduced shape:", X_pca5.shape)




print("Columns in X:", X.columns)



X_temp5, X_test5, y_temp5, y_test5 = train_test_split(
    X_pca5, y, test_size=0.2, random_state=42, stratify=y
)


# 87.5% of 0.8 = 0.7 train, 0.1 validation
X_train5, X_val5, y_train5, y_val5 = train_test_split(
    X_temp5, y_temp5, test_size=0.125, random_state=42, stratify=y_temp5
)



print("Train:", X_train5.shape, y_train5.shape)
print("Validation:", X_val5.shape, y_val5.shape)
print("Test:", X_test5.shape, y_test5.shape)

#%%


svm_model5 = SVC(probability=True, random_state=42)


best_svm5, best_params_svm = run_grid_search(svm_model5, param_grid_svm, X_train5, y_train5)


print("-----------------------------VALIDACIJA:")
val_metrics = evaluate_model(best_svm5, X_val5, y_val5)

print("----------------------------------TEST:")
test_metrics = evaluate_model(best_svm5, X_test5, y_test5)


#%%---------------------------------------------Logistic Regression ----------------------------------------

#deljenje na skupove za trening, validaciju i test
print("*********************************************************************\n")
print("**************  Logistic sa malim skupom*********************\n")
print("*********************************************************************")


param_grid_logreg = [
    {'penalty': ['l1'], 'C': [0.001, 0.01, 0.1, 1, 10], 'solver': ['liblinear', 'saga']},
    {'penalty': ['l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100], 'solver': ['lbfgs', 'liblinear', 'saga']}
]

X_lr= X[["Job Satisfaction","Career Change Interest","Salary","Job Opportunities","Age" ,"Years of Experience"]]


X_temp, X_test, y_temp, y_test = train_test_split(
    X_lr, y, test_size=0.2, random_state=42, stratify=y
)


# 87.5% of 0.8 = 0.7 train, 0.1 validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp
)


print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

#%%

logreg = LogisticRegression(max_iter=5000, random_state=42,class_weight="balanced")

# Run grid search
best_logreg, best_params_logreg = run_grid_search(logreg, param_grid_logreg, X_train, y_train)

print("-----------------------------VALIDACIJA:")
evaluate_model(best_logreg, X_val, y_val)

print("-----------------------------TEST:")
evaluate_model(best_logreg, X_test, y_test)


#%%---------------------------------------------Logistic Regression veci skup----------------------------------------

#deljenje na skupove za trening, validaciju i test
print("*********************************************************************\n")
print("**************  Logistic sa vecim skupom*********************\n")
print("*********************************************************************")



X_temp6, X_test6, y_temp6, y_test6 = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 87.5% of 0.8 = 0.7 train, 0.1 validation
X_train6, X_val6, y_train6, y_val6 = train_test_split(
    X_temp6, y_temp6, test_size=0.125, random_state=42, stratify=y_temp6
)


print("Train:", X_train6.shape, y_train6.shape)
print("Validation:", X_val6.shape, y_val6.shape)
print("Test:", X_test6.shape, y_test6.shape)

#%%

logreg6 = LogisticRegression(max_iter=5000, random_state=42,class_weight="balanced")

# Run grid search
best_logreg6, best_params_logreg6 = run_grid_search(logreg6, param_grid_logreg, X_train6, y_train6)

print("-----------------------------VALIDACIJA:")
evaluate_model(best_logreg6, X_val6, y_val6)

print("-----------------------------TEST:")
evaluate_model(best_logreg6, X_test6, y_test6)

#%%--------------------------------LOGISTIC SA LDA-----------------------------------------------------



#deljenje na skupove za trening, validaciju i test
print("*********************************************************************\n")
print("**************  Logistic sa LDA*********************\n")
print("*********************************************************************")



X_temp7, X_test7, y_temp7, y_test7 = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 87.5% of 0.8 = 0.7 train, 0.1 validation
X_train7, X_val7, y_train7, y_val7 = train_test_split(
    X_temp7, y_temp7, test_size=0.125, random_state=42, stratify=y_temp7
)

# za lda
# n_components = broj klasa - 1
lda = LDA(n_components=1)
X_train_lda = lda.fit_transform(X_train7, y_train7)
X_val_lda   = lda.transform(X_val7)
X_test_lda  = lda.transform(X_test7)


print("Train:", X_train_lda.shape, y_train7.shape)
print("Validation:", X_val_lda.shape, y_val7.shape)
print("Test:", X_test_lda.shape, y_test7.shape)

#%%

logreg7 = LogisticRegression(max_iter=5000, random_state=42,class_weight="balanced")

# Run grid search
best_logreg7, best_params_logreg6 = run_grid_search(logreg7, param_grid_logreg, X_train7, y_train7)

print("-----------------------------VALIDACIJA:")
evaluate_model(best_logreg7, X_val7, y_val7)

print("-----------------------------TEST:")
evaluate_model(best_logreg7, X_test7, y_test7)


#%%--------------------------------LOGISTIC SA PCA-----------------------------------------------------



#deljenje na skupove za trening, validaciju i test
print("*********************************************************************\n")
print("**************  Logistic sa PCA*********************\n")
print("*********************************************************************")



X_temp8, X_test8, y_temp8, y_test8 = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train8, X_val8, y_train8, y_val8 = train_test_split(
    X_temp8, y_temp8, test_size=0.125, random_state=42, stratify=y_temp8
)

pca = PCA(n_components=0.95)  # keep 95% variance
X_train_pca = pca.fit_transform(X_train8)  # fit only on train
X_val_pca   = pca.transform(X_val8)        # transform validation
X_test_pca  = pca.transform(X_test8)       # transform test

print("Original shape:", X.shape)
print("Reduced shape after PCA:", X_train_pca.shape)

#%%

logreg8 = LogisticRegression(max_iter=5000, random_state=42,class_weight="balanced")

best_logreg8, best_params_logreg8 = run_grid_search(logreg8, param_grid_logreg, X_train_pca, y_train)
print("Best params:", best_params_logreg)

# ---------------------- EVALUATION ----------------------
print("----------------------------- VALIDATION -----------------------------")
evaluate_model(best_logreg8, X_val_pca, y_val)

print("----------------------------- TEST -----------------------------")
evaluate_model(best_logreg8, X_test_pca, y_test)


