import sys
import pickle
sys.path.append("../tools/")
import pandas as pd
from sklearn.preprocessing import Imputer
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

def load_data_as_df():
    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)

    df = pd.DataFrame(data_dict)
    df = df.transpose()
    return df

df=load_data_as_df()

df = df.drop('email_address', axis=1)
df = df.astype(float)

df = df.drop("loan_advances", axis=1)
df = df.drop('restricted_stock_deferred', axis=1)
df = df.drop('director_fees', axis=1)
df = df.drop('deferral_payments', axis=1)
df = df.drop('deferred_income', axis=1)

df = df.drop('restricted_stock', axis=1)

for i in df.index:
        if df.ix[i].count() < 3:
            df = df.drop(i, axis=0)
            
df=df.drop('TOTAL',axis=0)
df=df.drop('THE TRAVEL AGENCY IN THE PARK',axis=0)

df['pct_from_poi'] = df['from_poi_to_this_person']/(df['to_messages'] + 1)
df['pct_to_poi'] = df['from_this_person_to_poi']/(df['from_messages'] + 1)

features_list=['bonus',
 'from_poi_to_this_person',
 'long_term_incentive',
 'shared_receipt_with_poi',
 'total_payments',
 'total_stock_value',
 'pct_from_poi',
 'pct_to_poi']

features = df[features_list]
labels = df['poi']
features_list.insert(0,'poi')
df2=df[features_list]
df1 = df2.transpose()
df1 = df1.to_dict()

from sklearn.cross_validation import train_test_split,StratifiedShuffleSplit
sss = StratifiedShuffleSplit(labels, 3, test_size=0.3, random_state=0)
    
for train_index, test_index in sss:
    features_train = features.iloc[train_index]
    features_test= features.iloc[test_index]
    labels_train, labels_test = labels.iloc[train_index], labels.iloc[test_index]
    
# from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.pipeline import Pipeline
clf_final = Pipeline([
        ('imp', Imputer(missing_values='NaN',strategy='median')),
        ('kbest', SelectKBest(f_classif,k=8)),
        ('clf', DecisionTreeClassifier(max_depth=9,max_features='log2',presort=False,criterion='entropy'))])

test_classifier(clf_final, df1,features_list, folds= 1000)

test_classifier(clf_final, df1,features_list, folds= 1000)

print test_classifier(clf_final, df1,features_list, folds = 1000)

dump_classifier_and_data(clf_final, df1, features_list)

