"""Identical-fold economic baselines, economic augmentation and service-score sensitivity."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from config import *
import numpy as np,pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from xgboost import XGBRegressor
import shap

u=pd.read_csv(SOURCE/'city_year_light_services.csv')
base=['brightness_z','density_z','year2015'];full=SATELLITE+['density_z','year2015']
d=u.dropna(subset=['service_z',*base,*full]).reset_index(drop=True)
assert len(d)==226 and d.city_key.nunique()==114
models={'Brightness':base,'Economic baseline':base+SES,'Satellite structure':full,'Joint structure and economics':full+SES}
pred=[];hyper=[];attr=[];pca_loadings=[]
for fold,(tr,te) in enumerate(GroupKFold(5).split(d,groups=d.city_key),1):
    a,b=d.iloc[tr],d.iloc[te]
    assert not set(a.city_key)&set(b.city_key)
    # A training-derived target PC is projected into held-out observations.
    pcscale=StandardScaler().fit(np.log(a[SERVICES]));pc=PCA(1).fit(pcscale.transform(np.log(a[SERVICES])))
    sign=1 if pc.components_[0].sum()>0 else -1
    train_pc=sign*pc.transform(pcscale.transform(np.log(a[SERVICES])))[:,0]
    test_pc=sign*pc.transform(pcscale.transform(np.log(b[SERVICES])))[:,0]
    psd=np.std(train_pc,ddof=1);train_pc/=psd;test_pc/=psd
    pca_loadings.append(dict(fold=fold,explained_variance=pc.explained_variance_ratio_[0],**dict(zip(SERVICES,sign*pc.components_[0]))))
    for target in ['Equal-weight service score','Training-fold principal component']:
        ya=a.service_z.to_numpy() if target.startswith('Equal') else train_pc
        yb=b.service_z.to_numpy() if target.startswith('Equal') else test_pc
        for name,features in models.items():
            losses=[]
            for alpha in [.1,1,10,100]:
                loss=[]
                for it,iv in GroupKFold(3).split(a,groups=a.city_key):
                    if target.startswith('Equal'):yt,yv=ya[it],ya[iv]
                    else:
                        ss=StandardScaler().fit(np.log(a.iloc[it][SERVICES]));pp=PCA(1).fit(ss.transform(np.log(a.iloc[it][SERVICES])))
                        sg=1 if pp.components_[0].sum()>0 else -1
                        yt=sg*pp.transform(ss.transform(np.log(a.iloc[it][SERVICES])))[:,0];sd=yt.std(ddof=1);yt/=sd
                        yv=sg*pp.transform(ss.transform(np.log(a.iloc[iv][SERVICES])))[:,0]/sd
                    f=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=alpha)).fit(a.iloc[it][features],yt)
                    loss.append(mean_squared_error(yv,f.predict(a.iloc[iv][features])))
                losses.append(np.mean(loss))
            alpha=[.1,1,10,100][int(np.argmin(losses))]
            f=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=alpha)).fit(a[features],ya);yp=f.predict(b[features])
            hyper.append(dict(fold=fold,target=target,model=name,alpha=alpha,inner_mse=min(losses)))
            pred.extend(dict(exposure_site_id=z.exposure_site_id,city_key=z.city_key,year=z.year,fold=fold,target=target,model=name,observed=float(y),predicted=float(v)) for z,y,v in zip(b.itertuples(),yb,yp))
    # Explain the joint model in municipalities that did not contribute to fitting.
    settings=[(1,5),(2,5),(2,10),(3,10)];losses=[];features=full+SES
    def boost(setting):return XGBRegressor(n_estimators=180,max_depth=setting[0],min_child_weight=setting[1],learning_rate=.035,subsample=1.,colsample_bytree=1.,reg_lambda=10.,objective='reg:squarederror',random_state=SEED,n_jobs=1)
    for setting in settings:
        ls=[]
        for it,iv in GroupKFold(3).split(a,groups=a.city_key):
            f=boost(setting).fit(a.iloc[it][features],a.iloc[it].service_z);ls.append(mean_squared_error(a.iloc[iv].service_z,f.predict(a.iloc[iv][features])))
        losses.append(np.mean(ls))
    setting=settings[int(np.argmin(losses))];f=boost(setting).fit(a[features],a.service_z);yp=f.predict(b[features]);ex=shap.TreeExplainer(f);sv=ex.shap_values(b[features])
    assert np.allclose(sv.sum(axis=1)+ex.expected_value,yp,atol=1e-5)
    for k,z in enumerate(b.itertuples()):
        pred.append(dict(exposure_site_id=z.exposure_site_id,city_key=z.city_key,year=z.year,fold=fold,target='Equal-weight service score',model='Joint structure and economics boosting',observed=z.service_z,predicted=float(yp[k])))
        attr.extend(dict(exposure_site_id=z.exposure_site_id,city_key=z.city_key,year=z.year,fold=fold,feature=c,value=float(b.iloc[k][c]),contribution=float(sv[k,j]),expected_value=float(ex.expected_value)) for j,c in enumerate(features))
    print('Completed municipal outer fold',fold,flush=True)
p=pd.DataFrame(pred);p.to_csv(RUN/'results/predictions.csv',index=False)
pd.DataFrame(hyper).to_csv(RUN/'results/prediction_hyperparameters.csv',index=False)
pd.DataFrame(attr).to_csv(RUN/'results/prediction_attributions.csv',index=False)
pd.DataFrame(pca_loadings).to_csv(RUN/'results/pca_service_loadings.csv',index=False)
metrics=[]
for (target,model),g in p.groupby(['target','model']):
    for fold,v in [('All',g)]+[(str(k),v) for k,v in g.groupby('fold')]:
        metrics.append(dict(target=target,model=model,fold=fold,n=len(v),municipalities=v.city_key.nunique(),r_squared=r2_score(v.observed,v.predicted),rmse=np.sqrt(mean_squared_error(v.observed,v.predicted)),mae=mean_absolute_error(v.observed,v.predicted)))
pd.DataFrame(metrics).to_csv(RUN/'results/prediction_metrics.csv',index=False)
draws=[];summary=[]
contrasts=[('Satellite structure','Brightness'),('Satellite structure','Economic baseline'),('Joint structure and economics','Economic baseline'),('Joint structure and economics boosting','Economic baseline')]
for target,g in p.groupby('target'):
    w=g.pivot(index=['exposure_site_id','year','city_key','fold','observed'],columns='model',values='predicted').reset_index();groups=w.city_key.unique();indices=[np.flatnonzero(w.city_key.eq(c)) for c in groups];rng=np.random.default_rng(SEED)
    for model,baseline in contrasts:
        if model not in w:continue
        est=np.sqrt(np.mean((w[model]-w.observed)**2))-np.sqrt(np.mean((w[baseline]-w.observed)**2));values=[]
        for b in range(2000):
            ix=np.concatenate([indices[i] for i in rng.integers(0,len(groups),len(groups))]);v=w.iloc[ix]
            value=np.sqrt(np.mean((v[model]-v.observed)**2))-np.sqrt(np.mean((v[baseline]-v.observed)**2));values.append(value)
            draws.append(dict(target=target,model=model,baseline=baseline,replicate=b+1,rmse_difference=value))
        summary.append(dict(target=target,model=model,baseline=baseline,rmse_difference=est,ci_low=np.quantile(values,.025),ci_high=np.quantile(values,.975),replicates=2000))
pd.DataFrame(draws).to_csv(RUN/'results/prediction_bootstrap.csv',index=False)
pd.DataFrame(summary).to_csv(RUN/'results/prediction_contrasts.csv',index=False)
print(pd.DataFrame(metrics).query("fold=='All'").to_string(index=False),flush=True)
