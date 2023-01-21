import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

class ModelSummary:
    def __init__(self, df, y):
        self.df = df
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df, self.y, test_size=0.3, random_state=42)
        
    def logistic_regression(self):
        logreg = LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=1000)
        logreg.fit(self.X_train, self.y_train)
        preds_is_lr = logreg.predict(self.X_train)
        preds_oos_lr = logreg.predict(self.X_test)
        accuracy_is_lr = accuracy_score(self.y_train, preds_is_lr)
        accuracy_oos_lr = accuracy_score(self.y_test, preds_oos_lr)
        precision_is_lr = precision_score(self.y_train, preds_is_lr)
        precision_oos_lr = precision_score(self.y_test, preds_oos_lr)
        recall_is_lr = recall_score(self.y_train, preds_is_lr)
        recall_oos_lr = recall_score(self.y_test, preds_oos_lr)
        performance_metrics = {'Model': ['Logistic_Regression','Logistic_Regression'],
                      'Accuracy': [accuracy_is_lr, accuracy_oos_lr],
                      'Precision': [precision_is_lr, precision_oos_lr],
                      'Recall': [recall_is_lr, recall_oos_lr]}
        Logistic_Regression_cm = pd.DataFrame(performance_metrics, index=['In-Sample', 'Out-of-Sample'])
        return Logistic_Regression_cm
    
    def random_forest(self):
        rf_clf = RandomForestClassifier(n_estimators=500, random_state=42, max_depth=5, 
                                        max_features='sqrt', class_weight = 'balanced_subsample')
        rf_clf.fit(self.X_train, self.y_train)
        preds_is_rf = rf_clf.predict(self.X_train)
        preds_oos_rf = rf_clf.predict(self.X_test)
        accuracy_is_rf = accuracy_score(self.y_train, preds_is_rf)
        accuracy_oos_rf = accuracy_score(self.y_test, preds_oos_rf)
        precision_is_rf = precision_score(self.y_train, preds_is_rf)
        precision_oos_rf = precision_score(self.y_test, preds_oos_rf)
        recall_is_rf = recall_score(self.y_train, preds_is_rf)
        recall_oos_rf = recall_score(self.y_test, preds_oos_rf)
        performance_metrics = {'Model': ['Random_Forest','Random_Forest'],
                      'Accuracy': [accuracy_is_rf, accuracy_oos_rf],
                      'Precision': [precision_is_rf, precision_oos_rf],
                      'Recall': [recall_is_rf, recall_oos_rf]}
        RF_cm = pd.DataFrame(performance_metrics, index=['In-Sample', 'Out-of-Sample'])
        return RF_cm

    def xgb(self):
        xg_clf = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, max_depth=7, 
                           learning_rate=0.1, min_child_weight=10,
                        scale_pos_weight=1, subsample=0.99)
        xg_clf.fit(self.X_train, self.y_train)
        preds_is_xgb = xg_clf.predict(self.X_train)
        preds_oos_xgb = xg_clf.predict(self.X_test)
        accuracy_is_xgb = accuracy_score(self.y_train, preds_is_xgb)
        accuracy_oos_xgb = accuracy_score(self.y_test, preds_oos_xgb)
        precision_is_xgb = precision_score(self.y_train, preds_is_xgb)
        precision_oos_xgb = precision_score(self.y_test, preds_oos_xgb)
        recall_is_xgb = recall_score(self.y_train, preds_is_xgb)
        recall_oos_xgb = recall_score(self.y_test, preds_oos_xgb)
        performance_metrics = {'Model': ['XGBoost','XGBoost' ],
                      'Accuracy': [accuracy_is_xgb, accuracy_oos_xgb],
                      'Precision': [precision_is_xgb, precision_oos_xgb],
                      'Recall': [recall_is_xgb, recall_oos_xgb]}
        XGB_cm = pd.DataFrame(performance_metrics, index=['In-Sample', 'Out-of-Sample'])
        return XGB_cm
    
    def get_summary(self):
        Logistic_Regression_cm = self.logistic_regression()
        RF_cm = self.random_forest()
        XGB_cm = self.xgb()
        model_summary = pd.concat([Logistic_Regression_cm, RF_cm,XGB_cm]).reset_index()
        model_summary.sort_values(['index','Model'])
        return model_summary