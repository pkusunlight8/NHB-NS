"""Observed dates, complete ADL states and fixed baseline social/city profiles."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from config import *
import numpy as np,pandas as pd
import argparse

parser=argparse.ArgumentParser(description='Prepare licensed participant records for the multistate fit.')
parser.add_argument('--intervals',type=Path,required=True)
parser.add_argument('--windows',type=Path,required=True)
parser.add_argument('--charls-core',type=Path,required=True)
parser.add_argument('--city-panel',type=Path,default=SOURCE/'city_year_light_services.csv')
args=parser.parse_args()
a=pd.read_csv(args.intervals,low_memory=False)
o=pd.read_csv(args.windows)
keep=['subject_id','wave0','wave1','interview_start_date','interview_end_date','observed_dt','death_lower_dt','death_upper_dt','death_interval_censored']
a=a.merge(o[keep],on=['subject_id','wave0','wave1'],validate='one_to_one')
core_path=args.charls_core
core=pd.read_stata(core_path,columns=['ID','rabyear','rabmonth','raeduc_c','raeducl','r2wtrespb','r3wtrespb'],convert_categoricals=False)
core['subject_id']=core.ID.astype(str).str.strip().str.replace(r'\.0$','',regex=True).map(lambda x:hashlib.sha1(x.encode()).hexdigest()[:16])
with pd.read_stata(core_path,iterator=True,convert_categoricals=False) as reader:
    metadata=reader.variable_labels();labels=reader.value_labels()
save_json(RUN/'qa/core_education_labels.json',{k:v for k,v in labels.items() if any(x in k.lower() for x in ['educ','edl','school'])})
a=a.merge(core.drop(columns='ID'),on='subject_id',how='left',validate='many_to_one')
u=pd.read_csv(args.city_panel)
cols=['exposure_site_id','year','service_z','service_relative_z','brightness_z','density_z',*SES]
context=u[cols].rename(columns={c:'baseline_'+c for c in cols if c not in ['exposure_site_id','year']})
b=a.sort_values(['subject_id','interview_start_date']).drop_duplicates('subject_id').copy()
b=b.merge(context,left_on=['city_site_id','exposure_year'],right_on=['exposure_site_id','year'],how='left',validate='many_to_one',suffixes=('','_context'))
b['baseline_female']=b.female;b['baseline_rural']=b.rural01
# CHARLS codes 1-4 cover no schooling through primary school; 5+ is middle school and above.
b['baseline_low_education']=b.raeduc_c.le(4).astype(float).where(b.raeduc_c.between(1,10))
b['baseline_education_level']=b.raeducl
b['baseline_age']=b.start_age
b['baseline_weight']=np.where(b.wave0.eq(2),b.r2wtrespb,b.r3wtrespb)
b['baseline_school_code']=b.raeduc_c
bcols=['subject_id']+[c for c in b if c.startswith('baseline_')]
a=a.merge(b[bcols],on='subject_id',how='left',validate='many_to_one')
a['start_month']=np.rint((a.interview_start_date*12)+.5).astype(int)
a['end_date']=a.interview_end_date
dead=a.event_is_death.eq(1)
a.loc[dead,'end_date']=a.loc[dead,'interview_start_date']+a.loc[dead,'death_upper_dt']
a['duration_months']=np.rint((a.end_date-a.interview_start_date)*12).astype(int)
a['birth_month_known']=a.rabmonth.between(1,12)
a['birth_month']=a.rabmonth.where(a.birth_month_known,6)
a['birth_ordinal']=a.rabyear*12+a.birth_month
a['exact_start_age']=(a.start_month-a.birth_ordinal)/12
a['exact_start_age']=a.exact_start_age.where(a.rabyear.between(1900,1975),a.start_age)
assert a.duration_months.gt(0).all()
assert not a.interval_id.duplicated().any()
profiles=['baseline_female','baseline_rural','baseline_low_education','baseline_service_z','baseline_brightness_z',*['baseline_'+c for c in SES]]
counts=[]
for endpoint in ['independence','multidomain']:
    d=a[a.start_age.between(60,100)].copy()
    counts.append(dict(endpoint=endpoint,step='Age 60-100',intervals=len(d),participants=d.subject_id.nunique()))
    if endpoint=='independence':
        d=d[d.start_six_complete.notna()&(d.end_six_complete.notna()|d.event_is_death.eq(1))].copy()
        d['origin']=d.start_six_complete.astype(int)
        d['destination']=np.where(d.event_is_death.eq(1),2,d.end_six_complete.fillna(0)).astype(int)
    else:
        d['origin']=d.start_state.ne(1).astype(int)
        d['destination']=np.where(d.event_is_death.eq(1),2,d.end_state.ne(1)).astype(int)
    counts.append(dict(endpoint=endpoint,step='Observed health states',intervals=len(d),participants=d.subject_id.nunique()))
    d=d.dropna(subset=profiles).copy()
    counts.append(dict(endpoint=endpoint,step='Complete baseline social and city profiles',intervals=len(d),participants=d.subject_id.nunique()))
    d.to_csv(RUN/f'analysis/{endpoint}_intervals.csv.gz',index=False)
    print(endpoint,len(d),d.subject_id.nunique(),d.city_site_id.nunique(),d.destination.value_counts().to_dict(),flush=True)
pd.DataFrame(counts).to_csv(RUN/'results/health_sample_accounting.csv',index=False)
b.drop(columns=[c for c in b if c in ['ID']]).to_csv(RUN/'analysis/baseline_profiles.csv.gz',index=False)
save_json(RUN/'qa/health_input_provenance.json',dict(inputs=[dict(filename=p.name,sha256=sha(p)) for p in [core_path,args.intervals,args.windows,args.city_panel]],death_observation='Absorbing status observed by end of report window. No death is assigned an exact date in the fitted observed-window likelihood.',birth_month_missing=int((~a.birth_month_known).sum()),baseline_school_codes=b.raeduc_c.value_counts().to_dict()))
