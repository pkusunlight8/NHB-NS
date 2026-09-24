"""Execute official IMaCh using fitted monthly laws and transformed covariance."""
from config import *
import numpy as np,pandas as pd,subprocess,itertools,argparse

def run(endpoint, exe):
    z=np.load(RUN/f'analysis/imach_{endpoint}/fit.npz');b=z['beta'];C=z['covariance']
    d=pd.read_csv(RUN/f'analysis/{endpoint}_intervals.csv.gz',low_memory=False)
    dest=RUN/f'analysis/native_{endpoint}';dest.mkdir(exist_ok=True)
    # Real surviving visit pairs supply age ranges for the calculator. Parameter
    # estimation, including all death windows, was completed in script 03.
    living=d[d.destination.lt(2)].copy();lines=[]
    for i,(_,r) in enumerate(living.iterrows(),1):
        # Source visit dates are decimal years, not Unix timestamps.
        sm=int(r.start_month);em=sm+int(r.duration_months)
        sy=(sm-1)//12;sy_month=(sm-1)%12+1;ey=(em-1)//12;ey_month=(em-1)%12+1
        birth_year=int(r.rabyear);birth_month=int(r.birth_month)
        assert 2013<=sy<=2015 and 2015<=ey<=2018
        lines.append(f'{i} {int(r.baseline_female)} 1 {birth_month}/{birth_year} 99/9999 {sy_month}/{sy} {int(r.origin)+1} {ey_month}/{ey} {int(r.destination)+1}')
    (dest/'observed_surviving_visits.txt').write_text('\n'.join(lines)+'\n',encoding='ascii')
    exe=Path(exe).resolve();statuses=[]
    for rural,low in itertools.product([0,1],repeat=2):
        stem=f'p{rural}{low}';T=np.zeros((12,40));x=np.array([0,rural,low,0,0,0,0,0])
        for t in range(4):T[3*t,t*10]=1;T[3*t,t*10+1]=-6;T[3*t,t*10+2:t*10+10]=x;T[3*t+1,t*10+1]=.1;T[3*t+2,t*10+2]=1
        bb=T@b.ravel();cc=T@C@T.T
        out=[f'title={stem} datafile=observed_surviving_visits.txt lastobs={len(living)} firstpass=1 lastpass=2',
         'ftol=1.e-08 stepm=1 ncovcol=1 nqv=0 ntv=0 nqtv=0 nlstate=2 ndeath=1 maxwav=2 mle=0 weight=0','model=1+age+V1.','#q','# Parameters']
        trans=['12','13','21','23']
        out += [t+' '+' '.join(f'{v:.14g}' for v in bb[3*i:3*i+3]) for i,t in enumerate(trans)]
        out += ['# Scales']+[f'{t} 0.0001 0.000001 0.0001' for t in trans]+['# Covariance matrix']
        for j in range(12):out.append(trans[j//3]+str(j%3+1)+' '+' '.join(f'{cc[j,i]:.14g}' for i in range(j+1)))
        out+=['agemin=60 agemax=90 bage=60 fage=90 estepm=1 ftolpl=1e-06','begin-prev-date=1/1/2013 end-prev-date=12/12/2018 mov_average=0','pop_based=0','prevforecast=0 yearsfproj=0 mobil_average=0','prevbackcast=0 yearsbproj=0 mobil_average=0','result:V1=0','result:V1=1']
        parameter=dest/(stem+'.imach');parameter.write_text('\n'.join(out)+'\n',encoding='ascii')
        with (dest/(stem+'.stdout.txt')).open('wb') as log:
            p=subprocess.run([str(exe),parameter.name],cwd=dest,stdout=log,stderr=subprocess.STDOUT,input=b'q\n',timeout=600,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),env={**os.environ,'TEMP':str(RUN/'tmp'),'TMP':str(RUN/'tmp')})
        statuses.append(dict(endpoint=endpoint,profile=stem,female=[0,1],rural=rural,low_education=low,returncode=p.returncode,parameter_sha256=sha(parameter)))
        print(statuses[-1],flush=True)
        if p.returncode!=0:raise RuntimeError((dest/(stem+'.stdout.txt')).read_text(errors='replace')[-3000:])
    save_json(RUN/f'qa/native_imach_{endpoint}.json',dict(version='0.99s14',executable_sha256=sha(exe),mode='mle=0; externally estimated observed-window parameters',profiles=statuses))
if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Convert fitted transition laws and run official IMaCh.')
    parser.add_argument('endpoint',choices=['independence','multidomain'])
    parser.add_argument('--imach-exe',type=Path,required=True)
    args=parser.parse_args()
    run(args.endpoint,args.imach_exe)
