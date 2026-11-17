"""Explicit-path entry points for independently prepared analysis inputs."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd


def json_value(x):
    if isinstance(x,dict):return {str(k):json_value(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [json_value(v) for v in x]
    if isinstance(x,np.ndarray):return json_value(x.tolist())
    if isinstance(x,(np.bool_,bool)):return bool(x)
    if isinstance(x,(np.integer,int)):return int(x)
    if isinstance(x,(np.floating,float)):return float(x) if np.isfinite(x) else None
    return x


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    fit=sub.add_parser('fit');fit.add_argument('--intervals',required=True);fit.add_argument('--arrays',required=True)
    fit.add_argument('--cluster-column',default='municipality_id');fit.add_argument('--maxiter',type=int,default=1000)
    fle=sub.add_parser('fle');fle.add_argument('--arrays',required=True)
    fle.add_argument('--start-age',type=float,default=60);fle.add_argument('--end-age',type=float,default=90);fle.add_argument('--initial-state',type=int,choices=[0,1],default=0)
    for name in ('reference','sparse'):
        sub.add_parser(name).add_argument('--input',required=True)
    allocation=sub.add_parser('allocation');allocation.add_argument('--table',required=True)
    allocation.add_argument('--score-columns',nargs='+',required=True);allocation.add_argument('--draws',type=int,default=2000)
    allocation.add_argument('--seed',type=int,default=20261103)
    spatial=sub.add_parser('spatial');spatial.add_argument('--features',required=True);spatial.add_argument('--arrays',required=True)
    spatial.add_argument('--radius',type=int,choices=[10000,25000],default=25000);spatial.add_argument('--caliper',type=float,choices=[.25,.5],default=.5)
    temporal=sub.add_parser('temporal');temporal.add_argument('--annual',required=True);temporal.add_argument('--weights',required=True);temporal.add_argument('--target-id',required=True)
    for parser in (fit,fle,*[sub.choices[n] for n in ['reference','sparse']],allocation,spatial,temporal):parser.add_argument('--output',required=True)
    args=p.parse_args();output=Path(args.output).expanduser().resolve()
    if output.exists():raise FileExistsError('Choose a new output path; existing files are not overwritten.')
    if args.command=='fit':
        from .clinical import fit_monthly_model
        table=pd.read_csv(args.intervals);a=np.load(args.arrays,allow_pickle=False)
        result=fit_monthly_model(table,a['design'],table[args.cluster_column].to_numpy(),maxiter=args.maxiter)
    elif args.command=='fle':
        from .clinical import restricted_expectancy,occupancy
        a=np.load(args.arrays,allow_pickle=False)
        kwargs=dict(start_age=args.start_age,end_age=args.end_age,initial_state=args.initial_state)
        result={'functional_years':restricted_expectancy(a['coefficients'],a['design'],**kwargs),'occupancy':occupancy(a['coefficients'],a['design'],**kwargs),'state_order':['independent','dependent','death'],'settings':kwargs}
    elif args.command=='reference':
        from .reference import ReferenceProblem,OperatingPoint,solve_reference
        d=json.loads(Path(args.input).read_text(encoding='utf-8'))
        fitted=solve_reference(ReferenceProblem(**d['problem']),point=OperatingPoint(**d.get('operating_point',{})),**d.get('options',{}))
        result={'decision':fitted.record(),'donor_ids':fitted.donor_ids,'weights':fitted.weights}
    elif args.command=='sparse':
        from .sparse_reference import exact_sparse
        d=json.loads(Path(args.input).read_text(encoding='utf-8'));problem=d['problem']
        for k in ['radiance','services','target_services','distance','d','t','point']:
            if k in problem:problem[k]=np.asarray(problem[k],float)
        result=exact_sparse(problem,d['arm'],int(d['k']),float(d['self_cap']))
    elif args.command=='allocation':
        from .inference import evaluate_city_table
        table=pd.read_csv(args.table,dtype={'city_id':str}).set_index('city_id')
        point,bootstrap,meta,alloc=evaluate_city_table(table,args.score_columns,draws=args.draws,seed=args.seed)
        meta.pop('resampled_city_counts')
        result={'point':point.to_dict('records'),'bootstrap':bootstrap.to_dict('records'),'metadata':meta,'city_order':table.index.tolist(),'allocations':alloc}
    elif args.command=='spatial':
        from .spatial import match_cells
        a=np.load(args.arrays,allow_pickle=False)
        result=match_cells(pd.read_csv(args.features),a['radiance'],a['coverage'],args.radius,args.caliper)
    else:
        from .temporal import fixed_reference_trajectory
        annual=pd.read_csv(args.annual,dtype={'city_id':str});weights=pd.read_csv(args.weights,dtype={'donor_id':str})
        trajectory,summary=fixed_reference_trajectory(annual,weights,args.target_id)
        result={'annual':trajectory.to_dict('records'),'summary':summary}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(json_value(result),indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':main()
