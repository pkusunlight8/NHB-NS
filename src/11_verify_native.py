"""Read actual IMaCh outputs and verify restricted state rewards."""
from config import *
import importlib.util,re,sys,numpy as np,pandas as pd
sp=importlib.util.spec_from_file_location('rewards',Path(__file__).with_name('07_expectancy.py'));mod=importlib.util.module_from_spec(sp);sys.modules[sp.name]=mod;sp.loader.exec_module(mod)
rows=[]
for endpoint in ['independence','multidomain']:
 beta=np.load(RUN/f'analysis/imach_{endpoint}/fit.npz')['beta']
 for rural in [0,1]:
  for low in [0,1]:
   stem=f'p{rural}{low}';file=RUN/f'analysis/native_{endpoint}/{stem}/E_{stem}.txt';sex=None;data={}
   for line in file.read_text().splitlines():
    m=re.search(r'V1=(\d)',line)
    if m:sex=int(m[1]);data[sex]={}
    if line.strip() and not line.startswith('#'):
     a=[float(x) for x in line.split()];data[sex][int(a[0])]=np.array([[a[1],a[2]],[a[4],a[5]]])
   for sex in [0,1]:
    pp=mod.profile(sex,rural,low)
    for age in [60,65,70,75,80,85]:
     for initial in [0,1]:
      expected,tr=mod.reward(beta,pp,float(age),float(90-age),initial)
      native=data[sex][age][initial]-tr[-1,:2]@data[sex][90]
      for j,name in enumerate(['favourable','unfavourable']):
       rows.append(dict(endpoint=endpoint,age=age,initial_state=initial,female=sex,rural=rural,low_education=low,metric=name,native_restricted_years=native[j],direct_integration_years=expected[j],absolute_difference=abs(native[j]-expected[j])))
d=pd.DataFrame(rows);d.to_csv(RUN/'results/native_imach_expectancies.csv',index=False)
assert d.absolute_difference.max()<.0002
save_json(RUN/'qa/native_imach_verification.json',dict(software='IMaCh 0.99s14',actual_runs=8,profile_sex_combinations=16,comparisons=len(d),max_absolute_difference_years=d.absolute_difference.max(),tolerance_years=.0002,passed=True))
print(d.groupby('endpoint').absolute_difference.agg(['count','max']).to_string())
