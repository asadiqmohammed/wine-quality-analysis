import matplotlib.pyplot as plt
import numpy as np
import missingno as msno
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
    data_dir = r'C:\Users\SADIQ\Desktop\Bachelors\Machine Learning\MIS\MIS\data'
    out_dir = './work/Labs/week4'
    file_name = 'winequality-red.csv'
    path = os.path.join(data_dir,file_name)

    df = pd.read_csv(path,encoding='ISO-8859-1',header=0)

    print("-------Wine File info----------")
    print(df.info())

    #---------------------------------------------------------------
    #  See missing data
    #---------------------------------------------------------------
    msno.matrix(df)
    plt.show()


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
    plt.show()
    # the alchol and acidity volitality affect correlation the most 
    # file_name = 'heatmap4-5.png'
    # path = os.path.join
    # path = os.path.join(out_dir,file_name)
    # plt.savefig(path)
    
    #---------------------------------------------------------------
    #  Identitfying quality (this here we range low and high quality then see how much content of what was included on average how much alchol,acid,vol acid,sulphates a column has )
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

    # (SelectKBest is builtin just ranks the importance of features and 
    #  part of prep work)

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


    #Separate data into training / testing sets (123321 as a seed for train_test_split we train and test so the model can train it self on some of the data the test itself with the remaining)
    
    XTrain,XTest,yTrain,yTest= train_test_split(X,y, random_state=seed)
    """
    (  fit_transform on training data:
    fit — the scaler looks at the training data and learns its mean and standard deviation for every feature.
    transform — it then applies that scaling to the training data.
    So it does two things at once — learn the scale, then apply it.
    transform only on test data: )
    """
    # Scale features (used to standardize my unit ranges)
    scaler        = StandardScaler()
    XTrain_scaled = scaler.fit_transform(XTrain)
    XTest_scaled  = scaler.transform(XTest)

    
    #---------------------------------------------------------------
    #  1st model linear regression
    #---------------------------------------------------------------
    """ ####################################
        # R2 Score — measures how much of the variation in quality the model explains (0 to 1)
        # 0.0 = explains nothing   0.5 = explains half   1.0 = perfect
        # Train vs Test gap reveals fit:
        #   both low + close    = underfit (model too simple)
        #   train high test low = overfit (model memorized training data)
        #   both high + close   = good fit
        # Our result: Train 0.38 / Test 0.27 — both low = underfit
        # wine quality has complex interactions between chemicals a straight line cant fully capture
        # low R2 here is not a failure — its a finding that tells us linear regression
        # is the wrong tool for this problem and points us toward trying classification models next
        transform only on test data: )
        ####################################
    """
    ## fitting training and testing linear
    Linear_reg_model = LinearRegression()
    Linear_reg_model.fit(XTrain,yTrain)
    Linear_reg_model_pred = Linear_reg_model.predict(XTest)

    print("\n------------------ Model 1: Linear Regression   ")
    print(Linear_reg_model.score(XTrain,yTrain))
    print(Linear_reg_model.score(XTest,yTest))
    print("\nNote: R2 of ~0.31 means that the model explains only 31% of variance in quality.")

    """ ####################################
    varinance refers to data spread)
    Your model is saying — out of all the reasons wines score differently from each other, I can account for 31% of those reasons using the chemical features.
    The other 69% of why wines score differently? The model has no explanation for it.
    ####################################
    """ 
    ###########
    ###########
    #########     LOOK AT UR RESPLOT TO SPEAK ON IT 
    """
    Residual plots are specific to regression models — they show how far off your predicted numbers are from the actual numbers on a continuous scale.
    Logistic Regression and KNN are classifiers — they predict categories (5, 6, 7) not continuous numbers. For classifiers you use a confusion matrix instead, which shows which categories got misclassified.
    Regression model  →  residual plot    (how far off was the number)
    Classification model  →  confusion matrix  (which category did it get wrong)
    """
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

    """
        # Classification Report
    # ─────────────────────────────────────────────────────
    # What it is:
    #   A breakdown of how well your classifier predicted each category individually
    #   instead of just giving one overall accuracy number
    # What it prints:
    #   precision  — of all the times the model predicted a score, how often was it right
    #                ex: predicted score 5 100 times, 69 were actually 5 = 0.69 precision
    #   recall     — of all the actual wines of that score, how many did the model catch
    #                ex: 168 actual score 5 wines, model correctly found 111 = 0.66 recall
    #   f1-score   — balance between precision and recall combined into one number
    #                0.0 = terrible   1.0 = perfect
    #   support    — how many wines of that score actually exist in the test set
    #                this is where you see the imbalance — score 5 has 168, score 3 has 5
    # How it is determined:
    #   compares your model's predictions against the actual known labels
    #   in the test set row by row and calculates each metric per class
    # What it is used for:
    #   overall accuracy alone is misleading with imbalanced data
    #   a model can be 61% accurate just by guessing 5 and 6 every time
    #   the report exposes which scores the model actually handles vs which it ignores
    #   in our case scores 3, 4, 8 show 0.00 across the board — model never predicts them
    #   that is the class imbalance problem made visible

    """
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
    plt.show()
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
    print("becase of class imbalance. 82.5% of wines are rated 5 or 6.")
    print("The models are not broken and working fine, but the data is limited")
    print("Our next should be to balance the data set before modeling")



####################### FINDINGS ########################