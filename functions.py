from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction import FeatureHasher


class Outlier_Drop_and_Skewness_handler(BaseEstimator, TransformerMixin):
    def __init__(self, features=None):
        if features is None:
            features = ['loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti', 'open_acc',
                        'revol_bal', 'revol_util', 'total_acc', 'Fico_average', 'mort_acc']
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        elif isinstance(X, np.ndarray):
            df = pd.DataFrame(X, columns=self.features)
        else:
            raise TypeError("Input must be a DataFrame or NumPy array.")

        not_present = [feat for feat in self.features if feat not in df.columns]
        if not_present:
            print(f"The following features are not in the dataframe: {not_present}")

        #for feat in self.features:
            #if df[feat].dtype.kind in 'bifc' and (df[feat] > 0).all():
                #df[feat] = np.log(df[feat] + 1)

        Q1 = df[self.features].quantile(0.25)
        Q3 = df[self.features].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df[self.features] < (Q1 - 3 * IQR)) | (df[self.features] > (Q3 + 3 * IQR))).any(axis=1)]

        return df
    

class features_to_drop(BaseEstimator, TransformerMixin):
    def __init__(self, feature=None):
        if feature is None:
            feature = ['funded_amnt_inv','funded_amnt','grade','emp_title',
                       'fico_range_high','fico_range_low','issue_d',
                       'title','addr_state','zip_code','earliest_cr_line']
        self.feature = feature

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        not_present = [feat for feat in self.feature if feat not in df.columns]
        if not_present:
            print(f"The following features are not in the dataframe: {not_present}")
        present_features = [feat for feat in self.feature if feat in df.columns]
        if present_features:
            df = df.drop(present_features, axis=1)
        return df
    

class one_hot_encoding(BaseEstimator, TransformerMixin):
    def __init__(self, n_features=10, feature=['home_ownership', 'verification_status', 'purpose', 'initial_list_status', 'application_type', 'region']):
        self.feature = feature
        self.hasher = FeatureHasher(n_features=n_features, input_type="string")

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        not_present = [feat for feat in self.feature if feat not in df.columns]
        if not_present:
            print(f"The following features are not in the dataframe: {not_present}")
        present_features = [feat for feat in self.feature if feat in df.columns]
        for feat in present_features:
            hashed_features = self.hasher.transform(df[feat].apply(lambda x: [x]).tolist())
            hashed_features = pd.DataFrame(hashed_features.toarray(), columns=[f"{feat}_hashed_{i}" for i in range(self.hasher.n_features)], index=df.index)
            df = pd.concat([df, hashed_features], axis=1)
            df = df.drop(feat, axis=1)
        return df
    

class FeatureHashing(BaseEstimator, TransformerMixin):
    def __init__(self, n_features=10, feature=['home_ownership', 'verification_status', 'purpose', 'initial_list_status', 'application_type', 'region']):
        self.feature = feature
        self.hasher = FeatureHasher(n_features=n_features, input_type="string")

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        not_present = [feat for feat in self.feature if feat not in df.columns]
        if not_present:
            print(f"The following features are not in the dataframe: {not_present}")
        present_features = [feat for feat in self.feature if feat in df.columns]
        for feat in present_features:
            hashed_features = self.hasher.transform(df[feat].apply(lambda x: [x]).tolist())
            hashed_features = pd.DataFrame(hashed_features.toarray(), columns=[f"{feat}_hashed_{i}" for i in range(self.hasher.n_features)], index=df.index)
            df = pd.concat([df, hashed_features], axis=1)
            df = df.drop(feat, axis=1)
        return df
    

class OrdinalFeatNames(BaseEstimator, TransformerMixin):
    def __init__(self, feature=['sub_grade']):
        self.feature = feature
        self.ordinal_enc = OrdinalEncoder()

    def fit(self, df, y=None):
        if (set(self.feature).issubset(df.columns)):
            self.ordinal_enc.fit(df[self.feature])
        else:
            print("One or more features are not in the dataframe")
        return self

    def transform(self, df):
        if (set(self.feature).issubset(df.columns)):
            df[self.feature] = self.ordinal_enc.transform(df[self.feature])
            return df
        else:
            print("One or more features are not in the dataframe")
            return df
        


class MinMaxWithFeatNames(BaseEstimator, TransformerMixin):
    def __init__(self, feature=['loan_amnt','int_rate','installment','emp_length','annual_inc','dti',
                                'open_acc','revol_bal','revol_util','total_acc', 'mort_acc',
                                'pub_rec_bankruptcies','Fico_average']):
        self.feature = feature
        self.min_max_enc = MinMaxScaler()

    def fit(self, df, y=None):
        if (set(self.feature).issubset(df.columns)):
            self.min_max_enc.fit(df[self.feature])
        else:
            print("One or more features are not in the dataframe")
        return self

    def transform(self, df):
        if (set(self.feature).issubset(df.columns)):
            df[self.feature] = self.min_max_enc.transform(df[self.feature])
            return df
        else:
            print("One or more features are not in the dataframe")
            return df
        
class Oversample(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        if 'loan_status' not in df.columns:
            raise ValueError("loan_status is not in the dataframe")

        # Separate the input features (X) and the target variable (y)
        X = df.loc[:, df.columns != 'loan_status']
        y = df['loan_status']

        # SMOTE function to oversample the minority class to fix the imbalanced data
        oversample = SMOTE(sampling_strategy='minority')
        X_bal, y_bal = oversample.fit_resample(X, y)

        # Concatenate the balanced X and y into a new dataframe
        df_bal = pd.concat([pd.DataFrame(X_bal, columns=X.columns), pd.DataFrame(y_bal, columns=['loan_status'])], axis=1)

        return df_bal
