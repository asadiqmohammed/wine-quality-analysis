import matplotlib.pyplot as plt
import numpy as np
# import missingno as msno
import os
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression,Ridge,SGDRegressor,LogisticRegression
from sklearn.metrics import  r2_score, mean_squared_error, mean_absolute_error,confusion_matrix,classification_report
from sklearn.feature_selection import SelectKBest,f_classif


if __name__ == '__main__': 
                        
    #------------------------
    #  Loading Data
    #------------------------
    seed = 123321
    path = os.path.join(os.path.dirname(__file__), 'winequality-red.csv')

    df = pd.read_csv(path,encoding='ISO-8859-1',header=0)

    print("-------Wine File info----------")
    print(df.info())

    #---------------------------------------------------------------
    #  See missing data
    #---------------------------------------------------------------
    # msno.matrix(df)
    # plt.show()


    #---------------------------------------------------------------
    # Data distrubuiton and imbalance check (she the value of quality and what numbers appears for it value)
    #---------------------------------------------------------------

    out_count = df['quality'].value_counts(normalize=True).sort_index()
    class_counts = df['quality'].value_counts().sort_index()

    print("------------------Quality Score Distribution ------------------")
    print(pd.DataFrame({'count': class_counts, 'percent': (out_count * 100).round(1)}))
    print(f"\nNOTE: Scores 5-6 is 82.5% of data therefore its a heavily imbalanced dataset and model will predict toward average quality")


    #---------------------------------------------------------------
    #  Correlation matrix - finding patterens
    #---------------------------------------------------------------
    corr = df.corr()

    #makes heat map and saves it 
    sns.set(font_scale=0.6)    
    sns.heatmap(corr, annot=True)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'heatmap.png'))
    plt.close()
    print("Heatmap saved to heatmap.png can be viewed")
    
    #---------------------------------------------------------------
    #  Identitfying quality 
    #---------------------------------------------------------------    

    # high quality (7-8) vs low quality (3-4) chemical profile comparison
    high = df[df['quality'] >= 7].mean().round(3)
    low  = df[df['quality'] <= 4].mean().round(3)
    profile = pd.DataFrame({'low_quality (3-4)': low, 'high_quality (7-8)': high})
    print("\n------------------ Chemical Profile: High vs Low Quality Wines ------------------")
    print(profile.drop('quality').to_string())
    print("\nNote: high quality wines have more alcohol (+1.3%), more sulphates, less volatile acidity")
    


    #---------------------------------------------------------------
    #  Feature selection(SelectKBest)
    #---------------------------------------------------------------    

    #load input out put set
    X = df.drop(['quality'],axis=1)
    y = df.loc[:,'quality']
    
    # SelectKBest
    kselect = SelectKBest(k='all', score_func=f_classif)
    X_best = kselect.fit_transform(X,y)
    feature_scores = pd.DataFrame({
        'feature': X.columns,
        'score':   kselect.scores_.round(2)
    }).sort_values('score', ascending=False)

    print("\n------------------ SelectKBest Feature Scores ------------------")
    print(feature_scores.to_string(index=False))
    print("\nNOTE: Alcohol (115.85) scores nearly double the next feature (volatile acidity 60.91),")
    print("independently confirming the correlation findings.")





    #---------------------------------------------------------------
    #  training/testing data 
    #---------------------------------------------------------------
    
    XTrain,XTest,yTrain,yTest= train_test_split(X,y, random_state=seed)
    # Scale features 
    scaler        = StandardScaler()
    XTrain_scaled = scaler.fit_transform(XTrain)
    XTest_scaled  = scaler.transform(XTest)

    
    #---------------------------------------------------------------
    #  1st model linear regression
    #---------------------------------------------------------------

    ## fitting training and testing linear
    Linear_reg_model = LinearRegression()
    Linear_reg_model.fit(XTrain,yTrain)
    Linear_reg_model_pred = Linear_reg_model.predict(XTest)

    print("\n------------------ Model 1: Linear Regression   ")
    print(Linear_reg_model.score(XTrain,yTrain))
    print(Linear_reg_model.score(XTest,yTest))
    print("\nNote: R2 of ~0.31 means that the model explains only 31% of variance in quality.")

    # Residual plot
    plt.figure(figsize=(8, 5))
    plt.scatter(Linear_reg_model_pred, yTest - Linear_reg_model_pred, alpha=0.5)
    plt.hlines(y=0, xmin=Linear_reg_model_pred.min(), xmax=Linear_reg_model_pred.max(), colors='red')
    plt.xlabel('Predicted Quality')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title('Linear Regression — Residual Plot')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'residual_plot.png'))
    plt.close()
    print("Res plot saved to residual_plot.png and can be viewed")


    #---------------------------------------------------------------
    #  2nd Model - logistic Regression
    #---------------------------------------------------------------

    log_model = LogisticRegression(max_iter=2000)
    log_model.fit(XTrain_scaled, yTrain)
    log_preds = log_model.predict(XTest_scaled)

    print("\n------------------ Model 2: Logistic Regression-----------------")
    print(f"Train Accuracy: {log_model.score(XTrain_scaled, yTrain):.4f}")
    print(f"Test  Accuracy: {log_model.score(XTest_scaled, yTest):.4f}")
    print("\nClassification Report:")
    print(classification_report(yTest, log_preds, zero_division=0))
    print("NOTE: Model predicts scores 5 and 6  well but fails on scores")
    print("3, 4, and 8 — directly caused by class imbalance.")

    #---------------------------------------------------------------
    #  3rd model KNN Classifier
    #---------------------------------------------------------------

    # fitting training and testing kneighbor
    knn_model = KNeighborsClassifier()
    knn_model.fit(XTrain, yTrain)
    knn_predict = knn_model.predict(XTest)


    print("\n------------------ Model 3: KNN Classifier ------------------")
    print(f"Train Accuracy: {knn_model.score(XTrain, yTrain):.4f}")
    print(f"Test  Accuracy: {knn_model.score(XTest, yTest):.4f}")
    print("\nClassification Report:")
    print(classification_report(yTest, knn_predict, zero_division=0))
    print("NOTE: Gap between train (0.65) and test (0.52) accuracy shows overfitting.")
    print("Just like Logistic Regression, KNN struggles on quality scores that are rare.")

    #Confusion Matrix
    cmat = confusion_matrix(yTest, knn_predict)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cmat, annot=True, fmt='d', cmap='Blues',
                xticklabels=sorted(y.unique()),
                yticklabels=sorted(y.unique()))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('KNN Classifier — Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'confusion_matrix.png'))
    plt.close()
    print("Confusion matrix saved to confusion_matrix.png can be viewed")



    #---------------------------------------------------------------
    #  Conclusion
    #---------------------------------------------------------------

    print("\n\n------------------ Conclusion ------------------")
    print("All three models underperform on rare quality scores 3, 4, 8 ")
    print("because of class imbalance. 82.5% of wines are rated 5 or 6.")
    print("The models are not broken and are working fine, but the data is limited.")
    print("Our next should be to balance the data set before modeling")



