"""Small invented mathematical fixtures. No study data are read or written."""
import unittest
import numpy as np
import pandas as pd
from lowradiance import clinical
from lowradiance.clinical_fast import loglik_scores
from lowradiance.horizon import fre_windows
from lowradiance.reference import ReferenceProblem,OperatingPoint,solve_reference
from lowradiance.sparse_reference import exact_sparse
from lowradiance.sparse_recovery import recover
from lowradiance.allocation import shares,project_share,oracle,loss
from lowradiance.inference import batch_project,evaluate_city_table
from lowradiance.unit_invariance import correct_predictions
from lowradiance.spatial import match_cells
from lowradiance.spatial_statistics import exact_p
from lowradiance.temporal import fixed_reference_trajectory


class NumericalInvariants(unittest.TestCase):
    def setUp(self):
        self.beta=np.zeros((4,5));self.beta[:,0]=[-4,-6,-3,-5]
        self.beta[:,1]=[.2,.3,-.1,.3]
        self.x=np.array([[1,0,0,0,0],[1,1,1,1,1]],float)

    def test_probability_conservation_and_absorption(self):
        p=clinical.transition_matrix(self.beta,self.x)
        np.testing.assert_allclose(p.sum(axis=2),1)
        np.testing.assert_array_equal(p[:,2,:],np.tile([0,0,1],(2,1)))
        states=clinical.occupancy(self.beta,self.x)
        np.testing.assert_allclose(states.sum(axis=2),1,atol=1e-12)

    def test_fast_likelihood_matches_interval_probability(self):
        frame=pd.DataFrame({'origin':[0,1],'destination':[1,2],'months':[12,24],
                            'death_lower':[0,12],'death_upper':[12,24],'age':[60,70]})
        probability=clinical.interval_probabilities(self.beta,frame,self.x)
        logs,scores=loglik_scores(self.beta,self.x,np.array([0,1]),np.array([1,2]),
                                 np.array([12,24]),np.array([0,12]),np.ones(2))
        np.testing.assert_allclose(logs,np.log(probability),atol=1e-12)
        step=1e-6;plus=self.beta.copy();minus=self.beta.copy();plus[0,0]+=step;minus[0,0]-=step
        numeric=(np.log(clinical.interval_probabilities(plus,frame,self.x))-np.log(clinical.interval_probabilities(minus,frame,self.x)))/(2*step)
        np.testing.assert_allclose(scores[:,0],numeric,rtol=1e-5,atol=1e-7)

    def test_functional_integral_and_horizon_agree(self):
        states=clinical.occupancy(self.beta,self.x,start_age=60,end_age=62)
        # Use age-60 designs consistently for this cross-kernel comparison.
        x=self.x.copy();x[:,1]=0
        end,integrals=fre_windows(self.beta[None],x,np.array([0,0]))
        expected=clinical.restricted_expectancy(self.beta,x,start_age=60,end_age=62)
        np.testing.assert_allclose(integrals[-1,0],expected[:,1],atol=1e-12)
        np.testing.assert_allclose(end[0],states[-1,:,1],atol=1e-12)
        self.assertTrue((np.diff(integrals[:,0],axis=0)>=0).all())

    def test_shapley_sum_identity(self):
        beta=self.beta.copy();beta[:,2:]=np.array([[.1,.2,.3],[-.2,.2,.1],[.1,-.1,.2],[.2,.1,-.1]])
        d=clinical.shapley_decomposition(beta,self.x[1],self.x[0],end_age=65)
        np.testing.assert_allclose(d['contributions'].sum(axis=0),d['difference'],atol=1e-10)

    def test_reference_matches_known_feasible_radiance_optimum(self):
        p=ReferenceProblem('target',['target','donor'],np.array([10.,5.]),10.,
                           np.array([[1.],[1.]]),np.array([1.]),
                           np.ones((8,2)),np.ones(8),np.array([0.,1.]))
        r=solve_reference(p,objective='radiance',risk_limit=1,point=OperatingPoint())
        self.assertTrue(r.feasible)
        self.assertAlmostEqual(r.achieved_gap,.5,places=6)
        self.assertLess(r.maximum_constraint_violation,1e-7)

    def sparse_problem(self):
        return dict(ids=['target','donor'],si=0,radiance=np.array([10.,5.]),target_radiance=10.,
                    services=np.array([[1.],[1.]]),target_services=np.array([1.]),distance=np.array([0.,1.]),
                    d=np.ones((8,2)),t=np.ones(8),task='long_30_year',band=0)

    def test_sparse_enumeration_and_recovery_agree(self):
        p=self.sparse_problem();a=exact_sparse(p,'FRE_long',1,1.);b=recover(p,1,1.)
        self.assertEqual(a['status'],'optimal');self.assertEqual(b['status'],'optimal')
        np.testing.assert_allclose(a['w'],b['w'],atol=1e-6)
        self.assertAlmostEqual(p['radiance']@a['w']/10,.5,places=6)

    def test_short_task_self_cap_infeasibility(self):
        p=self.sparse_problem();p['task']='short_24_month';p['radiance']=np.array([10.,6.])
        r=exact_sparse(p,'FRE_time',1,.25)
        self.assertEqual(r['status'],'infeasible')

    def test_budget_scale_invariance_and_oracle(self):
        q=np.array([1.,8.,2.]);c=np.array([4.,3.,3.])
        np.testing.assert_allclose(project_share(q,c),project_share(q*1000,c))
        a,minimum=oracle(q,c)
        self.assertAlmostEqual(loss(shares(q),a),minimum)
        np.testing.assert_allclose(batch_project(np.vstack([q,q*9]),np.vstack([c,c])),np.tile(a,(2,1)))

    def test_selected_rule_composition_is_unit_invariant(self):
        p=pd.DataFrame({'city_key':['a','b'],'baseline_mass':[1.,1.],
                        'class_fold':[0,1],'selected_traditional_rule':['r1','r2'],
                        'selected_traditional_risk':[1.,1.],'r1':[1.,4.],'r2':[8.,2.]})
        a=correct_predictions(p,['r1','r2'])[1].allocation_share
        p['r1']*=10000
        b=correct_predictions(p,['r1','r2'])[1].allocation_share
        np.testing.assert_allclose(a,b,atol=1e-12)

    def test_bootstrap_zero_event_draws_are_retained_in_denominator(self):
        table=pd.DataFrame({'baseline_mass':[1.,1.,1.],'event_mass':[1.,0.,0.],
                            'rule':[1.,1.,1.]},index=['a','b','c'])
        _,bootstrap,meta,_=evaluate_city_table(table,['rule'],draws=80,seed=9)
        self.assertEqual(meta['draws_defined']+meta['zero_event_draws'],80)
        self.assertEqual(len(bootstrap),meta['draws_defined'])
        self.assertGreater(meta['zero_event_draws'],0)

    def test_matching_excludes_ambiguous_ties_and_no_replacement(self):
        n=8;d=pd.DataFrame({k:np.ones(n) for k in ['population_total2020','population65plus2020',
                         'population65plus2020_coverage','population_total2020_coverage',
                         'hospital_drive_minutes2020','hospital_drive_minutes2020_coverage',
                         'elder_total_beds_per10002025','elder_total_beds_per10002025_coverage',
                         'nearest_bus_stop_m2024','tcc_baseline2012_2021','t2m_baseline2012_2021',
                         'tp_baseline2012_2021','sde_baseline2012_2021']})
        d['radius_m']=1000
        y=np.tile(np.arange(1,n+1,dtype=float),(14,1))
        r=match_cells(d,y,np.ones_like(y))
        self.assertEqual(r['matched_pairs'],2)
        self.assertEqual(len(set(r['control_indices'])),2)
        tied=match_cells(d,np.ones_like(y),np.ones_like(y))
        self.assertEqual(tied['matched_pairs'],0)
        self.assertEqual(tied['ambiguous_cells'],8)

    def test_temporal_requires_all_positive_donors(self):
        annual=pd.DataFrame([dict(city_id=city,year=year,radiance=value,valid=True)
                             for year in range(2012,2026)
                             for city,value in [('a',10.),('b',5.),('c',5.)]])
        weights=pd.DataFrame({'donor_id':['a','b','c'],'weight':[.5,.49,.01]})
        _,r=fixed_reference_trajectory(annual,weights,'a');self.assertTrue(r['stable'])
        annual.loc[(annual.city_id=='c')&(annual.year>=2022),'valid']=False
        trajectory,r=fixed_reference_trajectory(annual,weights,'a')
        self.assertFalse(r['stable']);self.assertEqual(r['validation_valid_years'],3)
        self.assertEqual(len(trajectory),14)

    def test_exact_sign_flip_known_tail(self):
        self.assertEqual(exact_p([1.,1.]),.5)
        self.assertEqual(exact_p([0.,0.]),1.)


if __name__=='__main__':unittest.main()
