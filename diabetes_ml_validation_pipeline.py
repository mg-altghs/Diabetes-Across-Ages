from pathlib import Path
import json, warnings, zipfile, os
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
 average_precision_score, confusion_matrix, brier_score_loss, balanced_accuracy_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
BASE=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('NHANES_DATA_DIR', BASE/'data'))
OUT=BASE/'results'; OUT.mkdir(exist_ok=True)
def read(stem): return pd.read_sas(ROOT/f'{stem}.xpt',format='xport')
D,Q,B,P=[read(x) for x in ['DEMO_L','DIQ_L','BMX_L','PAQ_L']]
df=D.merge(Q,on='SEQN',how='left',validate='one_to_one').merge(B,on='SEQN',how='left',validate='one_to_one').merge(P,on='SEQN',how='left',validate='one_to_one')
df=df[(df.RIDAGEYR>=18)&(df.DIQ010.isin([1,2]))].copy(); df['target']=(df.DIQ010==1).astype(int)
df['age_group']=np.where(df.RIDAGEYR<60,'Adults 18–59','Older adults 60+')
for c in ['DMDEDUC2','RIAGENDR','RIDRETH3']: df[c]=df[c].replace({7:np.nan,9:np.nan})
df['PAD680']=df['PAD680'].replace({7777:np.nan,9999:np.nan})
features=['RIDAGEYR','RIAGENDR','RIDRETH3','DMDEDUC2','INDFMPIR','BMXBMI','PAD680']
numeric=['RIDAGEYR','INDFMPIR','BMXBMI','PAD680']; categorical=['RIAGENDR','RIDRETH3','DMDEDUC2']
pre=ColumnTransformer([('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())]),numeric),('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),categorical)])
# 60/20/20 stratified split. Validation controls model/threshold; test remains untouched until final evaluation.
X=df[features]; y=df.target; ix=df.index
Xtemp,Xte,ytemp,yte,idtemp,idxte=train_test_split(X,y,ix,test_size=.20,random_state=42,stratify=y)
Xtr,Xval,ytr,yval,idxtr,idxval=train_test_split(Xtemp,ytemp,idtemp,test_size=.25,random_state=43,stratify=ytemp)
pos_weight=(ytr==0).sum()/(ytr==1).sum()
models={
 'Logistic Regression':LogisticRegression(max_iter=2500,class_weight='balanced'),
 'Random Forest':RandomForestClassifier(n_estimators=450,min_samples_leaf=5,max_features='sqrt',class_weight='balanced_subsample',random_state=42,n_jobs=-1),
 'XGBoost':XGBClassifier(n_estimators=450,max_depth=3,learning_rate=.035,subsample=.85,colsample_bytree=.85,reg_lambda=2.0,scale_pos_weight=pos_weight,objective='binary:logistic',eval_metric='logloss',random_state=42,n_jobs=-1)
}
def metrics(y,prob,thr):
 pred=(prob>=thr).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
 return {'threshold':float(thr),'accuracy':accuracy_score(y,pred),'precision':precision_score(y,pred,zero_division=0),'recall_sensitivity':recall_score(y,pred,zero_division=0),'specificity':tn/(tn+fp) if tn+fp else np.nan,'f1':f1_score(y,pred,zero_division=0),'balanced_accuracy':balanced_accuracy_score(y,pred),'roc_auc':roc_auc_score(y,prob),'pr_auc_average_precision':average_precision_score(y,prob),'brier_score':brier_score_loss(y,prob),'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp)}
rows=[]; fitted={}; validation_curves=[]
for name,est in models.items():
 pipe=Pipeline([('prep',pre),('model',est)]); pipe.fit(Xtr,ytr)
 pv=pipe.predict_proba(Xval)[:,1]
 # Threshold selected exclusively on validation set by F1; evaluate a transparent grid too.
 grid=np.arange(.05,.751,.01); vals=[f1_score(yval,(pv>=t).astype(int),zero_division=0) for t in grid]
 bestt=float(grid[int(np.argmax(vals))]); vm=metrics(yval,pv,bestt)
 rows.append({'model':name,**{f'validation_{k}':v for k,v in vm.items()}})
 fitted[name]=pipe
 for t in [.20,.30,.40,.50,bestt]:
  validation_curves.append({'model':name,**metrics(yval,pv,t)})
valdf=pd.DataFrame(rows).sort_values('validation_roc_auc',ascending=False); valdf.to_csv(OUT/'validation_model_selection.csv',index=False)
pd.DataFrame(validation_curves).to_csv(OUT/'validation_threshold_sensitivity.csv',index=False)
# Select model by validation ROC-AUC; threshold is its validation-F1 optimum.
chosen=valdf.iloc[0].model; pipe=fitted[chosen]; pv=pipe.predict_proba(Xval)[:,1]
grid=np.arange(.05,.751,.01); threshold=float(grid[np.argmax([f1_score(yval,(pv>=t).astype(int),zero_division=0) for t in grid])])
pt=pipe.predict_proba(Xte)[:,1]; test=metrics(yte,pt,threshold)
pd.DataFrame([{'selected_model':chosen,**test,'test_n':len(yte),'test_positive_n':int(yte.sum()),'test_prevalence':float(yte.mean()),'validation_selected_threshold':threshold}]).to_csv(OUT/'final_heldout_test_metrics.csv',index=False)
# Subgroup test metrics; descriptive only, with threshold fixed from validation.
sub=df.loc[idxte,['age_group']].copy(); sub['y']=yte; sub['probability']=pt; sub['prediction']=(pt>=threshold).astype(int)
age=[]
for grp,g in sub.groupby('age_group'):
 try: auc=roc_auc_score(g.y,g.probability)
 except: auc=np.nan
 m=metrics(g.y,g.probability,threshold)
 age.append({'age_group':grp,'n_test':len(g),'positive_n':int(g.y.sum()),'unweighted_prevalence':float(g.y.mean()),**m})
pd.DataFrame(age).to_csv(OUT/'age_stratified_final_test_metrics.csv',index=False)
# Overall and within-age-group permutation importance on held-out test set.
for label,mask in [('overall',np.ones(len(Xte),dtype=bool)),('Adults_18_59',df.loc[idxte,'age_group'].to_numpy()=='Adults 18–59'),('Older_adults_60_plus',df.loc[idxte,'age_group'].to_numpy()=='Older adults 60+')]:
 if mask.sum()<50: continue
 pi=permutation_importance(pipe,Xte.iloc[mask],yte.iloc[mask],n_repeats=10,random_state=42,scoring='roc_auc',n_jobs=-1)
 imp=pd.DataFrame({'feature':features,'importance_mean':pi.importances_mean,'importance_std':pi.importances_std,'subset':label}).sort_values('importance_mean',ascending=False)
 imp.to_csv(OUT/f'permutation_importance_{label}.csv',index=False)
# Held-out predictions without direct identifiers.
out=sub.copy(); out.to_csv(OUT/'heldout_test_predictions.csv',index=False)
metadata={'source':'NHANES August 2021–August 2023 (DEMO_L, DIQ_L, BMX_L, PAQ_L)','eligible_population':'Adults aged 18+ with definite self-reported diabetes diagnosis response (DIQ010=1 or 2)','eligible_n':len(df),'positive_n':int(y.sum()),'negative_n':int((1-y).sum()),'unweighted_analytic_sample_prevalence':float(y.mean()),'predictors':features,'target':'Self-reported clinician diagnosis of diabetes (DIQ010); 1=yes, 0=no','split':'Stratified 60% train / 20% validation / 20% held-out test; random states 42 and 43','model_selection':'Highest validation ROC-AUC; threshold selected on validation set by maximum F1; one final test evaluation','selected_model':chosen,'selected_threshold':threshold,'xgboost_scale_pos_weight':float(pos_weight),'excluded':['DIQ010 target','LBXGH HbA1c due to diagnostic leakage concern','SEQN respondent identifier','survey weights/design variables from predictive features'],'limitations':['Exploratory and unweighted classifier; not a national NHANES prevalence estimate','NHANES complex survey design not incorporated into model fitting/metrics','Single 2021–2023 cycle and no external validation','Self-reported diagnosed diabetes omits undiagnosed cases','Child participants excluded from classifier; model applies to adults only','Association/importance does not imply causation','Threshold optimized for F1 on validation data; different clinical costs imply a different threshold']}
(OUT/'run_metadata.json').write_text(json.dumps(metadata,indent=2))
print('SELECTED',chosen,'THRESHOLD',threshold,'N',len(df),'POS',int(y.sum()),'PREVALENCE',round(y.mean()*100,2))
print('\nVALIDATION\n',valdf.to_string(index=False))
print('\nFINAL TEST\n',pd.DataFrame([test]).to_string(index=False))
print('\nAGE\n',pd.DataFrame(age)[['age_group','n_test','positive_n','unweighted_prevalence','precision','recall_sensitivity','specificity','f1','roc_auc','pr_auc_average_precision']].to_string(index=False))
for f in sorted(OUT.glob('permutation_importance_*.csv')): print('\n',f.name,'\n',pd.read_csv(f).to_string(index=False))
# Bundle reproducible scripts, results, and source notes; source raw XPT files excluded to keep package lean.
