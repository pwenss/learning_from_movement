"""Offline stage recovery, migration, and paid Batch idempotency checks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import batch
import common
import stage_store
from run_city import database
import test_parallel as parallel_tests
import test_sampling_rounds as sampling_tests


class StageStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).parent,prefix='.qa_stage_')
        self.root=Path(self.tmp.name).resolve()
        self.patches=[patch.object(common,'SERVER_ROOT',self.root),
                      patch.object(stage_store,'CITY_IDS',['nyc_new_york','chicago']),
                      patch.dict('os.environ',{'OPENROUTER_API_KEY':'offline-test','STREETVIEW_NETWORK_GUARD':'0'})]
        for p in self.patches:p.start()
        (self.root/'state').mkdir()
        self.db=database(self.root)
        self.city='chicago';self.sid='chicago__000000'
        self.config={'dataset_id':'test','model':'test:batch','base_model':'test'}
        path=self.root/'work/chicago/start.request.json'
        common.save(path,{'custom_id':'rstart','body':{'messages':[]}})
        sample={'sample_id':self.sid,'city_id':self.city,'request_path':str(path),
                'start':{'links':[{'pano_id':'next'}]}}
        with self.db:
            self.db.execute('INSERT INTO samples VALUES (?,?,?,?,?)',
                            (self.sid,self.city,'candidates_ready',0,json.dumps(sample)))
            self.db.execute('INSERT INTO requests VALUES (?,?)',('rstart',self.sid))
            self.db.execute('INSERT INTO panos VALUES (?,?)',('reserved',self.sid))
            self.db.execute('INSERT INTO metadata VALUES (?,?)',('cache','{}'))
            self.db.execute('INSERT INTO api_attempts VALUES (?,?)',('cache',2))

    def tearDown(self):
        self.db.close()
        state=stage_store._states.pop(str(self.root),None)
        if state:state.raw.close()
        for p in reversed(self.patches):p.stop()
        self.tmp.cleanup()

    def migrate(self):
        self.db.close()
        stage_store.initialize(self.root)
        self.db=database(self.root)

    def restart(self):
        stage_store._states.pop(str(self.root)).raw.close()
        self.db=database(self.root)

    def inventory(self):
        return {str(p.relative_to(self.root)):(p.stat().st_mtime_ns,p.read_bytes())
                for p in self.root.rglob('*') if p.is_file()}

    def test_migration_and_point_updates_never_write_files(self):
        legacy=(self.root/'state/generation.sqlite').read_bytes()
        self.migrate()
        data=json.loads(self.db.execute('SELECT data FROM samples').fetchone()[0])
        self.assertEqual(data['request']['custom_id'],'rstart')
        self.assertEqual(self.db.execute('SELECT * FROM panos').fetchall(),[('reserved',self.sid)])
        self.assertEqual(self.db.execute('SELECT count(*) FROM metadata').fetchone()[0],0)
        before=self.inventory()
        for _ in range(100):
            with self.db:self.db.execute('UPDATE samples SET attempts=attempts+1')
        self.db.execute('INSERT INTO metadata VALUES (?,?)',('new','{}'))
        self.db.execute('INSERT INTO api_attempts VALUES (?,?)',('new',3))
        self.assertEqual(self.inventory(),before)
        stage_store.checkpoint(self.db,self.city,'candidates_completed')
        self.assertEqual((self.root/'state/generation.sqlite').read_bytes(),legacy)
        self.restart()
        self.assertEqual(self.db.execute('SELECT attempts FROM samples').fetchone()[0],100)
        self.assertEqual(self.db.execute('SELECT count(*) FROM metadata').fetchone()[0],0)
        self.assertEqual(self.db.execute('SELECT count(*) FROM api_attempts').fetchone()[0],0)

    def test_unfinished_stage_rolls_back_with_round_metadata(self):
        self.migrate()
        path=self.root/'state/acquisition_chicago.json'
        common.save(path,{'rounds':[1]})
        stage_store.checkpoint(self.db,self.city,'start_lookup_round_1')
        self.db.execute("UPDATE samples SET status='llm_selected'")
        common.save(path,{'rounds':[1,2]})
        self.restart()
        self.assertEqual(self.db.execute('SELECT status FROM samples').fetchone()[0],'candidates_ready')
        self.assertEqual(common.load(path),{'rounds':[1]})

    def test_accepted_batch_survives_restart_without_resubmission(self):
        self.migrate()
        response=Mock(status_code=202);response.json.return_value={'id':'paid-batch'}
        with patch.object(batch.requests,'post',return_value=response) as post:
            batch.Batch(self.root,self.db,self.config,self.city).submit_pending([self.sid])
            self.restart()
            self.assertFalse(batch.Batch(self.root,self.db,self.config,self.city).submit_pending([self.sid]))
            self.assertEqual(post.call_count,1)
        self.assertEqual(self.db.execute('SELECT provider_id FROM batches').fetchone()[0],'paid-batch')
        self.assertEqual(self.db.execute('SELECT attempts FROM samples').fetchone()[0],1)

    def test_unknown_batch_survives_restart_and_blocks_repeat(self):
        self.migrate()
        with patch.object(batch.requests,'post',side_effect=batch.requests.Timeout) as post:
            with self.assertRaises(RuntimeError):
                batch.Batch(self.root,self.db,self.config,self.city).submit_pending([self.sid])
            self.restart()
            with self.assertRaises(RuntimeError):
                batch.Batch(self.root,self.db,self.config,self.city).submit_pending([self.sid])
            self.assertEqual(post.call_count,1)

    def test_result_application_is_atomic_and_recoverable(self):
        self.migrate()
        accepted=Mock(status_code=202);accepted.json.return_value={'id':'paid-batch'}
        client=batch.Batch(self.root,self.db,self.config,self.city)
        with patch.object(batch.requests,'post',return_value=accepted):client.submit_pending([self.sid])
        response=Mock();response.json.return_value={'status':'completed','results':[{
            'custom_id':'rstart-a0','response':{'status_code':200,'body':{
                'model':'test','choices':[{'message':{'content':json.dumps({
                    'selected_index':0,'confidence':.9,'reason':'test'})}}]}}}]}
        original=client._apply_results
        def interrupt(*args):original(*args);raise RuntimeError('interrupted collect')
        with patch.object(batch.requests,'get',return_value=response),patch.object(client,'_apply_results',side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError,'interrupted'):client.collect()
        self.assertEqual(self.db.execute('SELECT status FROM samples').fetchone()[0],'candidates_ready')
        self.restart()
        with patch.object(batch.requests,'get',return_value=response):
            self.assertTrue(batch.Batch(self.root,self.db,self.config,self.city).collect())
        self.restart()
        sample=json.loads(self.db.execute('SELECT data FROM samples').fetchone()[0])
        self.assertEqual(len(sample['batch_history']),1)
        self.assertEqual(sample['decision']['selected_pano_id'],'next')

    def test_cleanup_keeps_referenced_images_and_batch_payload(self):
        self.migrate()
        directory=self.root/'work/chicago/public_images';directory.mkdir()
        keep=directory/('a'*64+'.jpg');orphan=directory/('b'*64+'.jpg')
        keep.write_bytes(b'accepted');orphan.write_bytes(b'interrupted')
        raw=json.loads(self.db.execute('SELECT data FROM samples').fetchone()[0])
        raw['candidate_files']=[str(keep)]
        self.db.execute('UPDATE samples SET data=?',(json.dumps(raw),))
        payload=self.root/'work/chicago/batch.json';common.save(payload,{})
        payload_image=directory/('c'*64+'.jpg');payload_image.write_bytes(b'pending batch')
        common.save(payload,{'requests':[{'body':{'messages':[{'content':[{
            'type':'image_url','image_url':{'url':'https://example.invalid/chicago/'+payload_image.name}}]}]}}]})
        self.db.execute('INSERT INTO batches VALUES (?,?,?,?,?)',
                        ('intent',self.city,'submitted','accepted',str(payload)))
        stage_store.checkpoint(self.db,self.city,'candidates_completed')
        stage_store.prune_candidates(self.db,self.root,self.city)
        self.assertTrue(keep.exists());self.assertFalse(orphan.exists());self.assertTrue(payload.exists())
        self.assertTrue(payload_image.exists())
        self.assertFalse(Path(raw['request_path']).exists())


class StageParallelTests(parallel_tests.ParallelTests):
    def setUp(self):
        self.cities_patch=patch.object(stage_store,'CITY_IDS',['nyc_new_york']);self.cities_patch.start()
        with patch('storage.connect',side_effect=stage_store.connect):
            super().setUp()

    def tearDown(self):
        stage_store._states.pop(str(self.root)).raw.close()
        self.cities_patch.stop()
        super().tearDown()

    def test_path_checkpoint_precedes_every_render(self):
        sample=self.insert(0)
        from test_invariants import node
        class FakeMaps:
            def __init__(self,*a,**k):pass
            def close(self):pass
            def get(self,request):return node(int(request['pano']))
        def render(*args):
            snapshot=common.load(self.root/'state/city_stages/nyc_new_york.json')
            self.assertEqual(snapshot['completed_stage'],'paths_completed')
            self.assertEqual(snapshot['sample_states'],{'navigation_accepted':1})
            self.assertEqual(len(snapshot['tables']['panos']),8)
        from run_city import finish_parallel
        with patch('maps.Maps',FakeMaps),patch('run_city.render_sample',side_effect=render):
            finish_parallel([sample['sample_id']],self.db,self.root,self.root,self.config,self.workers)
        self.assertEqual(self.db.execute('SELECT status FROM samples').fetchone()[0],'complete')


class StageSamplingTests(sampling_tests.SamplingRounds):
    def setUp(self):
        self.cities_patch=patch.object(stage_store,'CITY_IDS',['nyc_new_york']);self.cities_patch.start()
        with patch('storage.connect',side_effect=stage_store.connect):
            super().setUp()

    def tearDown(self):
        stage_store._states.pop(str(self.root)).raw.close()
        self.cities_patch.stop()
        super().tearDown()

    def test_crash_mid_query_reuses_entire_frozen_round(self):
        def interrupted(db,run,city,workers,radius):
            self.query(lambda i:i)(db,run,city,workers,radius)
            raise RuntimeError('crash before round checkpoint')
        with patch('run_city.lookup_parallel',side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError,'crash'):
                from run_city import acquire_starts
                acquire_starts(self.db,self.root,self.output,self.city,self.config,{'maps':2})
        before=common.load(self.root/'state'/f'samples_{self.city}.json')
        stage_store._states.pop(str(self.root)).raw.close();self.db=database(self.root)
        self.called.clear();state=self.acquire(lambda i:i)
        self.assertEqual(self.called,[0,1,2,3])
        self.assertEqual(len(state['rounds']),1)
        self.assertEqual(common.load(self.root/'state'/f'samples_{self.city}.json'),before)


if __name__=='__main__':unittest.main()
