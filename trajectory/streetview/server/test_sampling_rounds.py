"""Multi-round acquisition stops, spacing and resume, without external requests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import common
from run_city import acquire_starts, bind_start, database, lookup_start, summarize
from sampling import SAMPLING_POLICY, extend_city_round, sample_points, prepare
from test_invariants import node


class SamplingRounds(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).parent,prefix='.qa_')
        self.root=Path(self.tmp.name).resolve()
        self.patch=patch.object(common,'SERVER_ROOT',self.root)
        self.patch.start()
        (self.root/'state').mkdir()
        self.output=self.root/'output'
        self.db=database(self.root)
        self.city='nyc_new_york'
        self.feature={'type':'Feature','properties':{'id':self.city},'geometry':{
            'type':'Polygon','coordinates':[[[0,0],[.1,0],[.1,.1],[0,.1],[0,0]]]}}
        points,minimum,attempts=sample_points(self.feature,'seed',count=4)
        self.points=[{'sample_id':f'{self.city}__{i:06d}','city_id':self.city,'city_rank':1,
                      'name':'New York, USA','sample_index':i,**p} for i,p in enumerate(points)]
        common.save(self.root/'state'/f'boundary_{self.city}.json',self.feature)
        common.save(self.root/'state'/f'samples_{self.city}.json',{
            'seed':'seed','boundary_hash':common.digest(self.feature),'points':self.points,
            'minimum_distance_m':minimum,'attempts':attempts})
        self.policy={**SAMPLING_POLICY,'points_per_round':4,'target_start_panos':3}
        self.config={'sampling_policy':self.policy}
        self.called=[]

    def tearDown(self):
        self.db.close();self.patch.stop();self.tmp.cleanup()

    def query(self,success):
        def invoke(db,run,city,workers,radius):
            self.assertEqual(radius,100)
            rows=db.execute("SELECT data FROM samples WHERE status='sampled' ORDER BY id").fetchall()
            for raw, in rows:
                s=json.loads(raw);i=s['sample_index'];self.called.append(i)
                pano=success(i)
                if pano is None:
                    s['error']='ZERO_RESULTS';bind_start(db,s,'no_pano')
                else:
                    s['start']=node(pano);bind_start(db,s,'pano_bound')
        return invoke

    def acquire(self,success):
        with patch('run_city.lookup_parallel',side_effect=self.query(success)):
            return acquire_starts(self.db,self.root,self.output,self.city,self.config,{'maps':2})

    def test_radius_explicitly_100m(self):
        class Maps:
            def get(inner,request):
                self.assertEqual(request['radius'],100)
                return node(1)
        sample,status=lookup_start(Maps(),None,dict(self.points[0]))
        self.assertEqual(sample['start_search_radius_m'],100)
        self.assertEqual(status,'pano_bound')

    def test_first_round_stops_and_resume_never_queries_again(self):
        state=self.acquire(lambda i:i)
        self.assertTrue(state['complete']);self.assertEqual(state['stop_reason'],'target_reached')
        self.assertEqual(len(state['rounds']),1);self.assertEqual(state['sampled_points'],4)
        self.called.clear()
        self.acquire(lambda i:self.fail('Already frozen result must not be queried'))
        self.assertEqual(self.called,[])

    def test_second_round_threshold_uses_unique_not_returned_points(self):
        state=self.acquire(lambda i:0 if i<4 else i)
        self.assertEqual(len(state['rounds']),2)
        self.assertEqual(state['rounds'][0]['unique_start_panos'],1)
        self.assertEqual(state['unique_start_panos'],5)
        self.assertEqual(self.called,list(range(8)))
        self.assertEqual(self.db.execute("SELECT count(*) FROM samples WHERE status='duplicate_start_pano'").fetchone()[0],3)

    def test_three_round_cap_can_finish_below_target(self):
        state=self.acquire(lambda i:None)
        self.assertEqual(len(state['rounds']),3)
        self.assertEqual(state['sampled_points'],12)
        self.assertEqual(state['stop_reason'],'max_rounds_reached')
        self.assertFalse(state['target_reached'])
        self.assertEqual(self.called,list(range(12)))
        self.assertTrue(summarize(self.db,self.root,self.output,self.city))
        summary=common.load(self.output/'cities/01_nyc_new_york/summary.json')
        self.assertEqual(summary['terminal_failed'],12)
        self.assertTrue(summary['city_complete'])
        self.assertEqual(common.load(self.output/'summary.json')['frozen_points'],12)

    def test_resume_after_round_frozen_before_queries_preserves_points(self):
        calls=0
        def interrupt(db,run,city,workers,radius):
            nonlocal calls
            calls+=1
            if calls==2:raise RuntimeError('interrupt after round freeze')
            self.query(lambda i:None)(db,run,city,workers,radius)
        with patch('run_city.lookup_parallel',side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError,'interrupt'):
                acquire_starts(self.db,self.root,self.output,self.city,self.config,{})
        before=common.load(self.root/'state'/f'samples_{self.city}.json')['points']
        self.assertEqual(len(before),8)
        state=self.acquire(lambda i:i)
        self.assertEqual(len(state['rounds']),2)
        self.assertEqual(state['sampled_points'],8)
        self.assertEqual(self.called,list(range(8)))
        self.assertEqual(before,common.load(self.root/'state'/f'samples_{self.city}.json')['points'])

    def test_additional_round_spacing_and_ids_are_stable(self):
        previous=copy.deepcopy(self.points)
        points=extend_city_round(self.root,self.city,self.policy,2)
        self.assertEqual(points[:4],previous)
        self.assertEqual(points,extend_city_round(self.root,self.city,self.policy,2))
        self.assertEqual(len({s['sample_id'] for s in points}),8)
        self.assertTrue(all(common.distance(a,b)>100 for i,a in enumerate(points) for b in points[i+1:]))
        with self.assertRaises(ValueError):extend_city_round(self.root,self.city,self.policy,4)

    def test_prepare_new_policy_and_legacy_config_preservation(self):
        common.save(self.root/'code/trajectory/data/trajectory_routes.json',[{'id':self.city,'name':'New York, USA'}])
        with patch('sampling.SERVER_ROOT',self.root),patch('sampling.CITY_IDS',[self.city]):
            config,_=prepare(self.root,self.output,'dataset','seed')
            self.assertEqual(config['sampling_policy'],SAMPLING_POLICY)
            self.assertEqual(config['first_round_points_per_city'],1500)
            again,_=prepare(self.root,self.output,'dataset','seed')
            self.assertEqual(again,config)
            legacy={k:v for k,v in config.items() if k!='sampling_policy'}
            legacy['points_per_city']=legacy.pop('first_round_points_per_city')
            file=self.output/'dataset_config.json';common.save(file,legacy)
            before=file.read_bytes()
            old,_=prepare(self.root,self.output,'dataset','seed')
            self.assertEqual(old,legacy);self.assertEqual(file.read_bytes(),before)

    def test_legacy_started_city_does_not_mix_radius_or_add_rounds(self):
        for point in self.points:
            self.db.execute('INSERT INTO samples(id,city_id,status,data) VALUES (?,?,?,?)',
                            (point['sample_id'],self.city,'no_pano',json.dumps(point)))
        self.db.commit()
        with patch('run_city.lookup_parallel') as lookup:
            state=acquire_starts(self.db,self.root,self.output,self.city,{}, {})
        self.assertEqual(state['policy']['version'],1)
        self.assertIsNone(lookup.call_args.args[-1])
        self.assertEqual(len(state['rounds']),1)


if __name__=='__main__':unittest.main()
