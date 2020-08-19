
"""
Preprocessing
"""
import pandas as pd
from sklearn import preprocessing
from datetime import datetime
import joblib

df_train=pd.read_csv('../inputs/Train.csv')
df_test=pd.read_csv('../inputs/Test.csv')

full_df=pd.concat([df_train,df_test],axis=0).reset_index(drop=True)


cat_feat=['Offered_By', 'Category','Size', 'Price','Content_Rating',
          'Release_Version','OS_Version_Required']
for f in cat_feat:
    le=preprocessing.LabelEncoder()
    full_df.loc[:,f]=le.fit_transform(full_df[f].values)
    
full_df['Rating_grp']=pd.cut(full_df.Rating,[0,1,2,3,4,5])

def date_diff(date):
    date=date.split()
    date='-'.join(date)
    date=datetime.strptime(date, '%b-%d-%Y')
    today=datetime(2020, 8, 15)
    return (today-date).days
  
full_df['Recency']=full_df['Last_Updated_On'].apply(date_diff)

#output lebel encoder
le_out=preprocessing.LabelEncoder()
le_out.fit(df_train['Downloads'].values)
joblib.dump(le_out,'../models/le_out.joblib' )
full_df.loc[full_df['Downloads'].notnull(),'Downloads']=le_out.transform(full_df.loc[full_df['Downloads'].notnull(),'Downloads'])

full_df.loc[full_df.Downloads.isnull(),:].reset_index(drop=True).drop('Downloads',axis=1).to_csv('../inputs/test_pr.csv')
full_df.loc[full_df.Downloads.notnull(),:].reset_index(drop=True).to_csv('../inputs/train_pr.csv')
