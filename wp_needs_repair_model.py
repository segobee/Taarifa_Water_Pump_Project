#!/usr/bin/env python
# coding: utf-8


import pickle 

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score


# parameters

n_est=80
min_samples = 1
max_depth = 20


output_file = f'model_n_est={n_est}.bin'

# data preparation

water_train = pd.read_csv('water-train.csv')
water_label = pd.read_csv('water-label.csv')

df_one = pd.merge(water_train, water_label, on='id')

df_one['date_recorded'] = pd.to_datetime(df_one['date_recorded'])

strings = list(df_one.dtypes[df_one.dtypes == 'object'].index)


str_features_with_nan = ['funder', 'installer', 'subvillage', 'public_meeting', 
                         'scheme_management', 'scheme_name',  'permit']

str_features_with_no_missing_value = ['wpt_name', 'basin','region','lga','ward','recorded_by',
                                      'extraction_type','extraction_type_group','extraction_type_class',
                                      'management','management_group','payment','payment_type',
                                      'water_quality','quality_group','quantity','quantity_group',
                                      'source','source_type','source_class','waterpoint_type',
                                      'waterpoint_type_group','status_group']

# Looping through the values to remove the spaces with _
# First, I will turn every value to lower case before replacement of spaces
# This will be done on the features that have no missing values 

for col in str_features_with_no_missing_value:
    df_one[col] = df_one[col].astype(str).str.lower().str.replace(' ', '_')


# copy the dataframe 

df_new = df_one.copy()


df_new = df_new.dropna(subset=['subvillage'])


# filling the missing values

df_new['funder'].fillna('missing', inplace=True)
df_new['scheme_management'].fillna('missing', inplace=True)
df_new['installer'].fillna('missing', inplace=True)
df_new['public_meeting'].fillna('true', inplace=True)
df_new['permit'].fillna('true', inplace=True)


to_clean = ['funder', 'scheme_management', 'installer', 'public_meeting', 'permit', 'subvillage']
for col in to_clean:
    df_new[col] = df_new[col].astype(str).str.lower().str.replace(' ', '_')


df_new2 = df_new.copy()

# Apply log10 transformation
df_new2["amount_tsh"] = np.log10(df_new2["amount_tsh"] + 1)
df_new2["gps_height"] = np.log10(df_new2["gps_height"] + 1)
df_new2["num_private"] = np.log10(df_new2["num_private"] + 1)
df_new2["region_code"] = np.log10(df_new2["region_code"] + 1)
df_new2["district_code"] = np.log10(df_new2["district_code"] + 1)
df_new2["population"] = np.log10(df_new2["population"] + 1)


df_new2['gps_height'].fillna(0, inplace=True)


df_new2.replace([np.inf, -np.inf], 0, inplace=True)


df_new2['status_group'] = df_new2['status_group'].map({'functional': 2, 
                                                       'functional_needs_repair': 1, 
                                                       'non_functional': 0})


df_new3 = df_new2.copy()
df_new3 = df_new3[df_new3.water_quality != 'fluoride_abandoned'].reset_index(drop=True)
df_new3 = df_new3[df_new3.waterpoint_type_group != 'dam'].reset_index(drop=True)


drop_features = ['region', 'ward', 'payment_type', 'quantity', 'source', 'waterpoint_type', 
                 'extraction_type', 'extraction_type_group', 'management']

df_new3.drop(columns=drop_features, inplace=True)


group_1 = ['id','funder','installer','wpt_name','subvillage',
           'lga', 'scheme_management', 'scheme_name', 'status_group']

group_2 = ['id','basin', 'public_meeting', 'recorded_by', 'permit', 'management_group', 'payment',
           'water_quality', 'quality_group', 'quantity_group', 'source_type', 'source_class',
           'waterpoint_type_group', 'status_group']

df_new4 = df_new3.copy()

df_status_id = df_new4[['id', 'status_group']]

df_full_train, df_test = train_test_split(df_new4, test_size=0.2, random_state=11)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=11)

df_full_train = df_full_train.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


y_train = df_train.status_group.values
y_val = df_val.status_group.values
y_test = df_test.status_group.values


del df_train['status_group']
del df_val['status_group']
del df_test['status_group']


categorical_var1 = ['funder','installer','wpt_name','subvillage','lga','scheme_management']

categories = {}

for c in categorical_var1:
    categories[c] = list(df_new4[c].value_counts().head(10).index)

for c, values in categories.items():
    for v in values:
        df_train['%s_%s' % (c, v)] = (df_train[c] == v).astype('int') 
        df_val['%s_%s' % (c, v)] = (df_val[c] == v).astype('int')
        df_test['%s_%s' % (c, v)] = (df_test[c] == v).astype('int')

df_train.drop(columns=categorical_var1, inplace=True)
df_val.drop(columns=categorical_var1, inplace=True)
df_test.drop(columns=categorical_var1, inplace=True)

categorical_var2 = ['basin', 'public_meeting', 'permit', 'management_group', 
                    'payment', 'water_quality', 'quality_group', 'quantity_group', 
                    'source_type', 'source_class', 'waterpoint_type_group', 'extraction_type_class']

categories2 = {}

for c in categorical_var2:
    categories2[c] = list(df_new4[c].value_counts().index)

for c, values in categories2.items():
    for v in values:
        df_train['%s_%s' % (c, v)] = (df_train[c] == v).astype('int')  
        df_val['%s_%s' % (c, v)] = (df_val[c] == v).astype('int')
        df_test['%s_%s' % (c, v)] = (df_test[c] == v).astype('int')

df_train.drop(columns=categorical_var2, inplace=True)
df_val.drop(columns=categorical_var2, inplace=True)
df_test.drop(columns=categorical_var2, inplace=True)


drop_transform_classes = ['waterpoint_type_group_cattle_trough', 'waterpoint_type_group_improved_spring', 
                          'source_class_unknown', 'source_type_other', 'quantity_group_unknown', 
                          'quality_group_milky', 'quality_group_unknown', 'quality_group_fluoride', 
                          'quality_group_colored', 'water_quality_milky', 'water_quality_fluoride', 
                          'water_quality_coloured', 'water_quality_salty_abandoned', 'management_group_other', 
                          'management_group_unknown', 'scheme_management_other', 'installer_0']


df_train.drop(columns=drop_transform_classes, inplace=True)
df_val.drop(columns=drop_transform_classes, inplace=True)
df_test.drop(columns=drop_transform_classes, inplace=True)

x1 = df_train.iloc[:, [4,5]]
x2 = df_val.iloc[:, [4,5]]
x3 = df_test.iloc[:, [4,5]]

kmeans = KMeans(5)
kmeans.fit(x1)
kmeans.fit(x2)
kmeans.fit(x3)


identified_clusters1 = kmeans.fit_predict(x1)
identified_clusters2 = kmeans.fit_predict(x2)
identified_clusters3 = kmeans.fit_predict(x3)


df_train['loc_cluster'] = identified_clusters1
df_val['loc_cluster'] = identified_clusters2
df_test['loc_cluster'] = identified_clusters3


df_train.drop(columns=['longitude', 'latitude'], inplace=True)
df_val.drop(columns=['longitude', 'latitude'], inplace=True)
df_test.drop(columns=['longitude', 'latitude'], inplace=True)


clus_var = ['loc_cluster']
loc_classes = {}

for c in clus_var:
    loc_classes[c] = list(df_train[c].value_counts().index)


for c, values in loc_classes.items():
    for v in values:
        df_train['%s_%s' % (c, v)] = (df_train[c] == v).astype('int')  
        df_val['%s_%s' % (c, v)] = (df_val[c] == v).astype('int')
        df_test['%s_%s' % (c, v)] = (df_test[c] == v).astype('int')


df_train.drop(columns=clus_var, inplace=True)
df_val.drop(columns=clus_var, inplace=True)
df_test.drop(columns=clus_var, inplace=True)


df_train['recorded_by'] = df_train['recorded_by'].map({'geodata_consultants_ltd': 1})
df_val['recorded_by'] = df_val['recorded_by'].map({'geodata_consultants_ltd': 1})
df_test['recorded_by'] = df_test['recorded_by'].map({'geodata_consultants_ltd': 1})


df_train['status_group'] = y_train
df_val['status_group'] = y_val
df_test['status_group'] = y_test


full_df = pd.concat([df_train, df_val])
full_df = pd.concat([full_df, df_test])


full_df = full_df.reset_index(drop=True)

full_df['construction_year'] = full_df['construction_year'].map(
    {2013: 54, 2012: 53, 2011: 52, 2010: 51, 2009: 50, 2008: 49, 2007: 48, 2006: 47, 
     2005: 46, 2004: 45, 2003: 44, 2002: 43, 2001: 42, 2000: 41, 1999: 40, 1998: 39, 
     1997: 38, 1996: 37, 1995: 36, 1994: 35, 1993: 34, 1992: 33, 1991: 32, 1990: 31, 
     1989: 30, 1988: 29, 1987: 28, 1986: 27, 1985: 26, 1984: 25, 1983: 24, 1982: 23, 
     1981: 22, 1980: 21, 1979: 20, 1978: 19, 1977: 18, 1976: 17, 1975: 16, 1974: 15, 
     1973: 14, 1972: 13, 1971: 12, 1970: 11, 1969: 10, 1968: 9, 1967: 8, 1966: 7, 1965: 6,
     1964: 5, 1963: 4, 1962: 3, 1961: 2, 1960: 1}
)

const_not_0 = full_df[full_df['construction_year'].notna()]
const_is_0 = full_df[full_df['construction_year'].isna()]

const_not_0['construction_year'] = const_not_0.construction_year.astype(int)
const_is_0['construction_year'] = const_not_0.construction_year.astype(int)

const_not_0['construction_year'] = const_not_0.construction_year.astype('category')
const_is_0['construction_year'] = const_is_0.construction_year.astype('category')


const_is_0 = const_is_0.reset_index(drop=True)
const_not_0 = const_not_0.reset_index(drop=True)

y_const_is_0 = const_is_0.construction_year.values
y_const_not_0 = const_not_0.construction_year.values


df1_not_0 = const_not_0[['id','construction_year','date_recorded', 'scheme_name']]
df2_is_0 = const_is_0[['id','construction_year','date_recorded', 'scheme_name']]


feat_drop = ['id','construction_year','date_recorded', 'scheme_name']

X_const_is_0 = const_is_0.drop(columns=feat_drop)
X_const_not_0 = const_not_0.drop(columns=feat_drop)


status_var = ['status_group']
status_classes = {}

for c in status_var:
    status_classes[c] = list(X_const_is_0[c].value_counts().index)

for c, values in status_classes.items():
    for v in values:
        X_const_not_0['%s_%s' % (c, v)] = (X_const_not_0[c] == v).astype('int')  
        X_const_is_0['%s_%s' % (c, v)] = (X_const_is_0[c] == v).astype('int')


X_const_not_0.drop(columns=status_var, inplace=True)
X_const_is_0.drop(columns=status_var, inplace=True)


X = X_const_not_0
y = y_const_not_0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)

forest = RandomForestClassifier(n_estimators=45, max_depth=25, random_state=False, 
                                max_features=0.6, min_samples_leaf=3, n_jobs=-1)

forest.fit(X_train, y_train)

y_const_pred_train = forest.predict(X_train)
y_const_pred = forest.predict(X_test)

y_const_pred_proba = forest.predict_proba(X_test)

accuracy_train = accuracy_score(y_train, y_const_pred_train)
accuracy_test = accuracy_score(y_test, y_const_pred)


prediction = forest.predict(X_const_is_0)

X_const_is_0.insert(0, 'construction_year', prediction)


X_const_not_0['construction_year'] = y_const_not_0


df1_not_0 = df1_not_0.drop('construction_year', axis=1)
df2_is_0 = df2_is_0.drop('construction_year', axis=1)


df_merged_not_0 = pd.concat([df1_not_0, X_const_not_0], axis=1)
df_merged_is_0 = pd.concat([df2_is_0, X_const_is_0], axis=1)


frames = [df_merged_not_0, df_merged_is_0]
df_final = pd.concat(frames)


df_final['construction_year'] = df_final['construction_year'].map(
    {54: 2013, 53: 2012, 52: 2011, 51: 2010, 50: 2009, 49: 2008, 48: 2007, 47: 2006, 
     46: 2005, 45: 2004, 44: 2003, 43: 2002, 42: 2001, 41: 2000, 40: 1999, 39: 1998, 
     38: 1997, 37: 1996, 36: 1995, 35: 1994, 34: 1993, 33: 1992, 32: 1991, 31: 1990, 
     30: 1989, 29: 1988, 28: 1987, 27: 1986, 26: 1985, 25: 1984, 24: 1983, 23: 1982, 
     22: 1981, 21: 1980, 20: 1979, 19: 1978, 18: 1977, 17: 1976, 16: 1975, 15: 1974, 
     14: 1973, 13: 1972, 12: 1971, 11: 1970, 10: 1969, 9: 1968, 8: 1967, 7: 1966, 6: 1965,
     5: 1964, 4: 1963, 3: 1962, 2: 1961, 1: 1960}
)


df_final['recorded_year'] = df_final['date_recorded'].dt.year
df_final['water_pump_age'] = df_final['recorded_year'] - df_final['construction_year']
df_final = df_final.drop(columns=['construction_year', 'recorded_year', 'date_recorded'])


df_final = df_final.reset_index(drop=True)

df_final = df_final.drop('scheme_name', axis=1)

df_final_updated = pd.merge(df_final, df_status_id, on='id')

df_final_updated = df_final_updated.drop(columns=['id', 'status_group_2', 'status_group_0', 'status_group_1'])


df_final_updated['status_group'] = df_final_updated['status_group'].map({2: 'functional', 
                                                                         1: 'functional_needs_repair', 
                                                                         0: 'non_functional'})


df_final_updated['status_group'] = df_final_updated['status_group'].astype('category')


X = df_final_updated.drop('status_group', axis=1)
y = df_final_updated.status_group.values


label_encoder=LabelEncoder()
label_encoder.fit(y)
y=label_encoder.transform(y)
classes=label_encoder.classes_


X_full_train, X_test, y_full_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_full_train, y_full_train, test_size=0.25, random_state=1)

train_dicts = X_train.to_dict(orient='records')
val_dicts = X_val.to_dict(orient='records')
test_dicts = X_test.to_dict(orient='records')

dv = DictVectorizer(sparse=False)

X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)
X_test = dv.transform(test_dicts)

dv = DictVectorizer(sparse=False)
train_dicts = X_full_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)


# training function

print('training the model')

def train(df_train, y_train, n_est=80):
    train_dicts = X_full_train.to_dict(orient='records')
    
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_dicts)
    
    rf_model=OneVsRestClassifier(RandomForestClassifier(n_estimators=n_est,
                                                  min_samples_leaf=min_samples,
                                                  max_depth=max_depth,
                                                  random_state=1))
    rf_model.fit(X_train,y_full_train)
    
    return dv, rf_model

# predicting function

def predict(df, dv, rf_model):
    
    X = dv.transform(test_dicts)
    y_pred = rf_model.predict_proba(X)
    
    return y_pred


dv, rf_model = train(X_full_train, y_train, n_est=80)

y_pred = predict(X_test, dv, rf_model)


y_test_binarized=label_binarize(y_test,classes=np.unique(y_test))

auc = roc_auc_score(y_test_binarized, y_pred)

print(f'auc={auc}')


 
# Saving the model

# I will be using pickle to save the model 

# Opening and closing the model file 

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, rf_model), f_out)

print(f'the model is saved to {output_file}')

