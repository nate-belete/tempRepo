import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler

class Data_Preprocessing:
    
    def __init__(self, data):
        self.data = data
    
    def convert_categorical_columns(self):
        self.data = self.data.astype({
            'has_spouse': 'category',
            'demographic': 'category',
            'region': 'category',
            'education_level': 'category',
            'kp_business_line': 'category',
            'competitor': 'category',
            'switched_to_hdhp': 'category',
            'kp_offer_hmo': 'int',
            'kp_offer_dhmo': 'int',
            'comp_offer_hmo': 'int',
            'comp_offer_dhmo': 'int',
            'comp_offer_hdhp': 'int'
        })
        return self.data
    
    def missing_rows_and_columns(self):
        cols_to_impute = ['med_risk_score', 'total_risk_score', 'has_spouse',
                                               'avg_household_income', 'kp_tenure_days', 
                          'education_level', 'demographic']
        for col in cols_to_impute:
            self.data[f'imputed_{col}'] = self.data[col].isna()
        return self.data
    
    def impute_mode(self, col_to_impute):
        # calculate the mode of the column
        mode = self.data[col_to_impute].mode()[0]
        # replace the missing values with the mode
        self.data[col_to_impute].fillna(mode, inplace=True)
        return self.data
    
    def impute_IterativeImputer(self):
        cols_with_missing = [col for col in self.data.columns if self.data[col].isnull().any()]
        df_with_missing = self.data[cols_with_missing].copy()
        imputer = IterativeImputer(random_state=0)
        imputed_data = imputer.fit_transform(df_with_missing)
        imputed_df = pd.DataFrame(imputed_data, columns=cols_with_missing)
        self.data[cols_with_missing] = imputed_df
        return self.data
    
    def scale_numerical_variables(self, numerical_vars):
        scaler = MinMaxScaler()
        self.data[numerical_vars] = scaler.fit_transform(self.data[numerical_vars])
        return self.data
    
    def get_dataset_properties(self):
        table = {
            'Rows': self.data.shape[0],
            'Columns': self.data.shape[1],
            '% Duplicate Rows': self.data.duplicated().mean() * 100,
            'Target Column': 'switched_to_hdhp',
            '% Missing Target Values': self.data['switched_to_hdhp'].isnull().mean() * 100
        }
        return pd.DataFrame(table, index=[0])
    
    def detected_column_types(self):
        # Get the data types of the variables
        data_types = self.data.dtypes

        # Create a dataframe to store the data types
        type_table = pd.DataFrame(columns = ['Variable Type', 'Count', 'Percentage'])

        # Get the counts of the different data types
        numeric_count = (data_types == 'int64').sum() + (data_types == 'float64').sum()
        categorical_count = (data_types == 'object').sum()
        text_count = 0
        datetime_count = 0
        sequence_count = 0
        total_count = numeric_count + categorical_count + text_count + datetime_count + sequence_count

        # Add the counts to the table
        types = ['Numeric', 'Categorical', 'Text', 'Datetime', 'Sequence']
        counts = [numeric_count, categorical_count, text_count, datetime_count, sequence_count]
        percentages = [numeric_count/total_count, categorical_count/total_count, 
                       text_count/total_count, datetime_count/total_count, sequence_count/total_count]

        type_table['Variable Type'] = types
        type_table['Count'] = counts
        type_table['Percentage'] = percentages

        # Display the table
        return type_table
    
    def corr_data(self):
        corr_data = self.data.copy()
        corr_data['switched_to_hdhp'] = corr_data['switched_to_hdhp'].factorize()[0]

        # get the most predictive numeric features
        numeric_features = corr_data.select_dtypes(include=[ 'int']).columns.tolist()

        # get the correlation matrix for the most predictive features
        corr_matrix = corr_data[numeric_features].corr()

        # show the full correlation matrix
        corr_matrix = corr_matrix.sort_values('switched_to_hdhp', ascending = False)

        # plot the correlation matrix
        sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', 
                    xticklabels=corr_matrix.columns.values,
                    yticklabels=corr_matrix.columns.values,
                    )
        plt.show()

        return None
    
    def One_Hot_Encode(self, cols):
        for col in cols:
            self.data = pd.get_dummies(self.data, columns=[col])
        return self.data
    
    
    def EDA_Plots(self):
        # plotting all figures in a single figure
        fig = plt.figure(figsize=(20,25))
        axs = [fig.add_subplot(5,4,i+1) for i in range(18)]
        plot_configs = [
            {'title': 'Switched to HDHP', 'x':'switched_to_hdhp', 'hue': None},
            {'title': 'Has Spouse', 'x': 'has_spouse', 'hue': 'switched_to_hdhp'},
            {'title': 'Med Risk Score', 'x': 'switched_to_hdhp', 'hue': None, 'y': 'med_risk_score'},
            {'title': 'Total Risk Score', 'x': 'switched_to_hdhp', 'hue': None, 'y': 'total_risk_score'},
            {'title': 'Age', 'x': 'switched_to_hdhp', 'hue': None, 'y': 'age'},
            {'title': 'Number of Dependents', 'x': 'switched_to_hdhp', 'hue': None, 'y': 'num_dependents'},
            {'title': 'Avg Household Income', 'x': 'switched_to_hdhp', 'hue': None, 'y': 'avg_household_income'},
            {'title': 'KP Tenure Days', 'x': 'switched_to_hdhp', 'hue': None, 'y': 'kp_tenure_days'},
            {'title': 'Education Level', 'x': 'education_level', 'hue': 'switched_to_hdhp'},
            {'title': 'KP Offer HMO', 'x': 'kp_offer_hmo', 'hue': 'switched_to_hdhp'},
            {'title': 'KP Offer DHMO', 'x': 'kp_offer_dhmo', 'hue': 'switched_to_hdhp'},
            {'title': 'Comp Offer HMO', 'x': 'comp_offer_hmo', 'hue': 'switched_to_hdhp'},
            {'title': 'Comp Offer DHMO', 'x': 'comp_offer_dhmo', 'hue': 'switched_to_hdhp'},
            {'title': 'Comp Offer HDHP', 'x': 'comp_offer_hdhp', 'hue': 'switched_to_hdhp'},
            {'title': 'KP Business Line', 'x': 'kp_business_line', 'hue': 'switched_to_hdhp'},
            {'title': 'Competitor', 'x': 'competitor', 'hue': 'switched_to_hdhp'},
            {'title': 'Region', 'x': 'region', 'hue': 'switched_to_hdhp'},
            {'title': 'Demographic', 'x': 'demographic', 'hue': 'switched_to_hdhp'},
        ]
        
        for i, ax in enumerate(axs):
            plot_config = plot_configs[i]
            title = plot_config.get('title')
            x = plot_config.get('x')
            hue = plot_config.get('hue')
            y = plot_config.get('y')
            if y is None:
                sns.countplot(x=x, hue=hue, data=self.data, ax=ax)
            else:
                sns.boxplot(x=x, y=y, hue=hue, data=self.data, ax=ax)
            ax.set_title(title)
        plt.tight_layout()
        plt.show()
        
        return None
    
    def Missing_Features(self):
        plt.figure(figsize=(15,6))
        self.data.isnull().sum()[self.data.isnull().sum() > 0].sort_values().plot.barh()
        plt.title('Missing Values Count')
        plt.xlabel('Variables')
        plt.ylabel('Count of Missing Values')
        plt.show()