
"""
model catboost
"""

import pandas as pd
import numpy as np
from sklearn import metrics,preprocessing
import dispatcher
from catboost import Pool


df=pd.read_csv("../inputs/train_pr_f.csv").drop('Last_Updated_On',axis=1)

features=[f for f in df.columns if f not in ['Downloads','kfold']]

##feature scaling
sc=preprocessing.MinMaxScaler()
sc.fit(df.loc[:,['Rating','Reviews','Recency']])
df.loc[:,['Rating','Reviews','Recency']]=sc.transform(df.loc[:,['Rating','Reviews','Recency']])
fold=1

cat_feat=['Offered_By', 'Category','Size', 'Price','Content_Rating',
          'Release_Version','OS_Version_Required','Rating_grp']

for f in cat_feat:
    df[f]=df[f].astype(str)
    
    
# =============================================================================
# df_train=df[df.kfold!=fold].reset_index(drop=True)
# df_valid=df[df.kfold==fold].reset_index(drop=True)
# 
# x_train=df_train[features]
# y_train=df_train.Downloads
# x_valid=df_valid[features]
# y_valid=df_valid.Downloads.values
# 
# 
#     
# train_pool=Pool(data=x_train,label=y_train,cat_features=cat_feat,feature_names=x_train.columns.tolist())
# test_pool=Pool(data=x_valid,label=y_valid,cat_features=cat_feat,feature_names=x_valid.columns.tolist())
# 
# =============================================================================

X=df[features]
y=df.Downloads
train_pool=Pool(data=X,label=y,cat_features=cat_feat,feature_names=X.columns.tolist())

model=dispatcher.models['catboost']
#model.fit(train_pool,eval_set=test_pool)
model.fit(train_pool)

test=pd.read_csv('../inputs/test_pr.csv').drop(['Unnamed: 0','Last_Updated_On'],axis=1)
test.loc[:,['Rating','Reviews','Recency']]=sc.transform(test.loc[:,['Rating','Reviews','Recency']])

for f in cat_feat:
    test[f]=test[f].astype(str)
    
y_pred=model.predict_proba(test)
y_pred=pd.DataFrame(y_pred)

y_pred.to_csv("../inputs/sub_cat2.csv",index=False)

