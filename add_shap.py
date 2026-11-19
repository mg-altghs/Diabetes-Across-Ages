import sys, json
from pathlib import Path
import os
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import shap
BASE=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('NHANES_DATA_DIR', BASE/'data')); OUT=BASE/'results'; OUT.mkdir(exist_ok=True)
def read(s): return pd.read_sas(ROOT/f'{s}.xpt',format='xport')
D,Q,B,P=[read(x) for x in ['DEMO_L','DIQ_L','BMX_L','PAQ_L']]
df=D.merge(Q,on='SEQN',how='left').merge(B,on='SEQN',how='left').merge(P,on='SEQN',how='left')
df=df[(df.RIDAGEYR>=18)&df.DIQ010.isin([1,2])].copy(); y=(df.DIQ010==1).astype(int)
for c in ['DMDEDUC2','RIAGENDR','RIDRETH3']: df[c]=df[c].replace({7:np.nan,9:np.nan})
df['PAD680']=df.PAD680.replace({7777:np.nan,9999:np.nan})
features=['RIDAGEYR','RIAGENDR','RIDRETH3','DMDEDUC2','INDFMPIR','BMXBMI','PAD680']; num=['RIDAGEYR','INDFMPIR','BMXBMI','PAD680']; cat=['RIAGENDR','RIDRETH3','DMDEDUC2']
pre=ColumnTransformer([('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),cat)])
X=df[features]; ix=df.index
Xtemp,Xte,ytemp,yte,idtemp,idxte=train_test_split(X,y,ix,test_size=.20,random_state=42,stratify=y)
Xtr,Xval,ytr,yval,idxtr,idxval=train_test_split(Xtemp,ytemp,idtemp,test_size=.25,random_state=43,stratify=ytemp)
pipe=Pipeline([('prep',pre),('model',LogisticRegression(max_iter=2500,class_weight='balanced'))]); pipe.fit(Xtr,ytr)
Xt=pipe.named_steps['prep'].transform(Xtr.sample(n=min(1500,len(Xtr)),random_state=42))
# Linear SHAP on the transformed design matrix; report grouped source-variable importance.
explainer=shap.LinearExplainer(pipe.named_steps['model'],Xt)
Xtest=pipe.named_steps['prep'].transform(Xte.sample(n=min(1200,len(Xte)),random_state=43))
sv=explainer.shap_values(Xtest)
if isinstance(sv,list): sv=sv[-1]
try: names=pipe.named_steps['prep'].get_feature_names_out()
except: names=[f'feature_{i}' for i in range(sv.shape[1])]
absmean=np.abs(sv).mean(axis=0)
raw=[]
for nm,val in zip(names,absmean):
 rawname=nm.split('__',1)[-1]
 for f in features:
  if rawname==f or rawname.startswith(f+'_'):
   raw.append((f,float(val))); break
imp=pd.DataFrame(raw,columns=['feature','mean_absolute_shap_value']).groupby('feature',as_index=False).sum().sort_values('mean_absolute_shap_value',ascending=False)
imp.to_csv(OUT/'shap_global_importance.csv',index=False)
print(imp.to_string(index=False))
