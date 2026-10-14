"""Offline invariants: no API calls and no server files outside test root."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import common
from common import bearing, dates_valid, distance, validate_route
from batch import Batch, parse_decision
from run_city import database, search
from sampling import sample_points


def node(i):
    return {'pano_id':str(i),'lat':40.0,'lon':-74.0+i*0.0001,'date':'2025-01',
            'source_pano_heading':90.,'source_pano_world_size':{'width':16384,'height':8192},
            'source_pano_tile_size':{'width':512,'height':512},
            'links':[{'pano_id':str(i+1),'heading':90.}]}


class Invariants(unittest.TestCase):
    def setUp(self):
        base=Path(__file__).resolve().parent
        self.tmp=tempfile.TemporaryDirectory(dir=base,prefix='.qa_')
        self.root=Path(self.tmp.name)
        self.patch=patch.object(common,'SERVER_ROOT',self.root)
        self.patch.start()
        (self.root/'state').mkdir()
        self.db=database(self.root)

    def tearDown(self):
        self.db.close()
        self.patch.stop()
        self.tmp.cleanup()

    def test_path_escape_refused(self):
        with self.assertRaises(ValueError):
            common.confined(self.root/'..'/'outside')

    def test_sampling_reproducible_separated_and_hole_free(self):
        feature={'type':'Feature','properties':{'id':'test'},'geometry':{'type':'Polygon','coordinates':[
            [[0,0],[.1,0],[.1,.1],[0,.1],[0,0]],
            [[.04,.04],[.06,.04],[.06,.06],[.04,.06],[.04,.04]]]}}
        a,minimum,_=sample_points(feature,'fixed',count=60)
        b,_,_=sample_points(feature,'fixed',count=60)
        self.assertEqual(a,b)
        self.assertGreater(minimum,100)
        self.assertTrue(all(not(.04<p['lat']<.06 and .04<p['lon']<.06) for p in a))

    def test_sampling_does_not_relax_spacing(self):
        feature={'properties':{'id':'tiny'},'geometry':{'type':'Polygon','coordinates':[
            [[0,0],[.0001,0],[.0001,.0001],[0,.0001],[0,0]]]}}
        with self.assertRaises(RuntimeError):
            sample_points(feature,'fixed',count=2,max_attempts=100)

    def test_dates_allow_later_year_precision_but_not_final_mixed_months(self):
        nodes=[{'date':'2025-01'},{'date':'2025-02'}]
        self.assertTrue(dates_valid(nodes))
        self.assertFalse(dates_valid(nodes,True))
        self.assertTrue(dates_valid(nodes+[{'date':'2025'}],True))
        self.assertFalse(dates_valid([{'date':'2025-13'}]))

    def test_llm_index_not_bool_or_fraction_and_confidence_finite(self):
        for index,confidence in [(True,.8),(.5,.8),(2,.8),(0,float('nan'))]:
            body={'choices':[{'message':{'content':json.dumps({'selected_index':index,'confidence':confidence,'reason':'x'})}}]}
            with self.assertRaises(ValueError):
                parse_decision(body,2)

    def sample(self):
        sample={'sample_id':'test__000000','start':node(0),'decision':{'selected_pano_id':'1'}}
        self.db.execute('INSERT INTO samples(id,city_id,status,data) VALUES (?,?,?,?)',
                        (sample['sample_id'],'test','llm_selected',json.dumps(sample)))
        self.db.commit()
        return sample

    def test_search_fixed_first_hop_and_no_ninth_fetch(self):
        sample=self.sample()
        class Maps:
            seen=[]
            def get(self,request):
                self.seen.append(request)
                return node(int(request['pano']))
        maps=Maps()
        frames,error=search(sample,maps,self.db,{'max_expansions':2000,'search_seconds':300})
        self.assertIsNone(error)
        self.assertEqual(len(frames),8)
        self.assertNotIn({'pano':'8'},maps.seen)
        validate_route(frames,'0','1')
        invalid=copy.deepcopy(frames)
        invalid[-1]['next_pano_id']='8'
        with self.assertRaises(ValueError):
            validate_route(invalid,'0','1')

    def test_global_start_conflict_before_api(self):
        sample=self.sample()
        self.db.execute('INSERT INTO panos VALUES (?,?)',('0','previous'))
        self.db.commit()
        frames,error=search(sample,None,self.db,{'max_expansions':2000,'search_seconds':300})
        self.assertIsNone(frames)
        self.assertEqual(error,'global_pano_conflict')

    def test_search_budget_persists(self):
        sample=self.sample()
        sample['search_budget']={'seconds':301,'expansions':0}
        class Maps:
            def get(self,request): return node(int(request['pano']))
        frames,error=search(sample,Maps(),self.db,{'max_expansions':2000,'search_seconds':300})
        self.assertEqual(error,'search_budget_exceeded')
        self.assertGreaterEqual(json.loads(self.db.execute('SELECT data FROM samples').fetchone()[0])['search_budget']['seconds'],301)

    def test_batch_unknown_submission_never_reposted(self):
        import requests
        with patch.dict('os.environ',{'OPENROUTER_API_KEY':'test-value'}):
            batch=Batch(self.root,self.db,{'model':'openai/gpt-5.6-luna:batch'},'test')
        with patch('batch.requests.post',side_effect=requests.Timeout):
            with self.assertRaisesRegex(RuntimeError,'submission_unknown'):
                batch.submit([{'custom_id':'r1','body':{}}])
        with patch('batch.requests.get') as get:
            with self.assertRaisesRegex(RuntimeError,'Unresolved'):
                batch.collect()
            get.assert_not_called()

    def test_batch_shuffled_results_and_missing_item_preserve_success(self):
        config={'model':'openai/gpt-5.6-luna:batch','base_model':'openai/gpt-5.6-luna'}
        items=[]
        for i in range(3):
            sid=f'test__{i:06d}'
            sample={'sample_id':sid,'start':{'links':[{'pano_id':'next'}]}}
            self.db.execute('INSERT INTO samples(id,city_id,status,data) VALUES (?,?,?,?)',(sid,'test','candidates_ready',json.dumps(sample)))
            self.db.execute('INSERT INTO requests VALUES (?,?)',(f'r{i}',sid))
            items.append({'custom_id':f'r{i}','body':{}})
        payload=self.root/'payload.json'
        common.save(payload,{'requests':items})
        self.db.execute('INSERT INTO batches VALUES (?,?,?,?,?)',('b','test','submitted','provider-b',str(payload)))
        self.db.commit()
        body={'model':config['base_model'],'choices':[{'message':{'content':json.dumps({'selected_index':0,'confidence':.8,'reason':'x'})}}]}
        response=unittest.mock.Mock()
        response.json.return_value={'status':'completed','results':[
            {'custom_id':'r2','response':{'status_code':200,'body':body}},
            {'custom_id':'r0','response':{'status_code':200,'body':body}}]}
        with patch.dict('os.environ',{'OPENROUTER_API_KEY':'test'}):
            batch=Batch(self.root,self.db,config,'test')
        with patch('batch.requests.get',return_value=response):
            self.assertTrue(batch.collect())
        outcomes=dict(self.db.execute('SELECT id,status FROM samples'))
        self.assertEqual(outcomes,{'test__000000':'llm_selected','test__000001':'candidates_ready','test__000002':'llm_selected'})
        with patch('batch.requests.get') as get:
            self.assertTrue(batch.collect())
            get.assert_not_called()

    def test_bad_first_hop_does_not_switch_to_alternate(self):
        sample=self.sample()
        sample['start']['links'].append({'pano_id':'alternate','heading':90.})
        class Maps:
            seen=[]
            def get(self,request):
                self.seen.append(request['pano'])
                target=node(1)
                target['lon']+=.01
                return target
        maps=Maps()
        frames,error=search(sample,maps,self.db,{'max_expansions':2000,'search_seconds':300})
        self.assertEqual(error,'llm_selected_nonviable_link')
        self.assertNotIn('alternate',maps.seen)

    def test_batch_configuration_rejection_stops_without_item_repairs(self):
        config={'model':'openai/gpt-5.6-luna:batch','base_model':'openai/gpt-5.6-luna'}
        payload=self.root/'payload.json'
        common.save(payload,{'requests':[{'custom_id':'r0','body':{}}]})
        self.db.execute('INSERT INTO batches VALUES (?,?,?,?,?)',('b','test','submitted','provider-b',str(payload)))
        self.db.commit()
        response=unittest.mock.Mock()
        response.json.return_value={'status':'failed','error':{'message':'Unsupported image transport'},
                                    'results':[],'request_counts':{'completed':0,'failed':1}}
        with patch.dict('os.environ',{'OPENROUTER_API_KEY':'test'}):
            batch=Batch(self.root,self.db,config,'test')
        with patch('batch.requests.get',return_value=response),patch('batch.requests.post') as post:
            with self.assertRaisesRegex(RuntimeError,'configuration rejected'):
                batch.collect()
            post.assert_not_called()
        self.assertEqual(self.db.execute('SELECT status FROM batches').fetchone()[0],'provider_validation_error')


if __name__=='__main__':
    unittest.main()
