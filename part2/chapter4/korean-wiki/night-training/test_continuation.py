"""Small orchestration/identity tests; no training or model construction."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import continue_run as c
import summarize


class ContinuationTests(unittest.TestCase):
    def test_stale_pre_stop_validation_is_not_used_as_current_weight_reference(self):
        old={'consumed':10,'updates':2,'validation':{'loss':9},'generation':['old']}
        fresh={'consumed':20,'updates':4,'validation':{'loss':3},'generation':['new']}
        reference,matched=c.evaluation_reference([old],{'consumed':20,'updates':4},fresh)
        self.assertIs(reference,fresh);self.assertFalse(matched)
        reference,matched=c.evaluation_reference([old,fresh],{'consumed':20,'updates':4},dict(fresh))
        self.assertIs(reference,fresh);self.assertTrue(matched)

    def test_missing_fresh_resume_evaluation_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'actual resumed'):
            c.evaluation_reference([],{'consumed':20,'updates':4},None)

    def test_reused_pid_or_different_run_is_not_reported_as_our_process(self):
        run=Path('/work/owned-run');pid=123
        commands=[(f'{pid} 1024 200.0 01:02 {c.BASE}/run.py --output {run}',True),
                  (f'{pid} 1024 200.0 01:02 {c.BASE}/run.py --output /work/other-run',False),
                  (f'{pid} 1024 200.0 01:02 /usr/bin/something-else',False)]
        for line,expected in commands:
            with self.subTest(expected=expected,line=line),patch.object(c.subprocess,'run',return_value=subprocess.CompletedProcess([],0,line,'')):
                result=c.process_snapshot(pid,run);self.assertEqual(result['process_identity_confirmed'],expected)
                self.assertEqual(result['process_rss_bytes'],1048576 if expected else None)

    def test_failed_process_observation_is_unresolved_not_terminal(self):
        with patch.object(c.subprocess,'run',return_value=subprocess.CompletedProcess([],2,'','temporary ps failure')):
            with self.assertRaises(OSError):c.process_snapshot(123,Path('/work/owned-run'))

    def test_custom_control_refuses_live_trainer_or_missing_terminal_record(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory).resolve();control=run/'control.json'
            c.write(control,{'run_dir':str(run),'pid':123,'exit_code':None});c.write(run/'progress.json',{'pid':123})
            with patch.object(summarize.subprocess,'run',return_value=subprocess.CompletedProcess([],1,'','')):
                with self.assertRaisesRegex(RuntimeError,'terminal'):summarize.assert_not_running(run,control)
            c.write(control,{'run_dir':str(run),'pid':123,'exit_code':0})
            with patch.object(summarize.subprocess,'run',return_value=subprocess.CompletedProcess([],0,f'python {c.BASE}/run.py --output {run}','')):
                with self.assertRaisesRegex(RuntimeError,'alive'):summarize.assert_not_running(run,control)
            with patch.object(summarize.subprocess,'run',return_value=subprocess.CompletedProcess([],1,'','')):
                self.assertEqual(summarize.assert_not_running(run,control)['exit_code'],0)

    def test_archive_preserves_exact_evaluated_state_and_does_not_invent_next_epoch(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory);info={'identity':{'source':'fixture'},'config':{'fixture':True},'evaluation_window_sha256':'fixed'}
            metric={'label':'epoch_2','kind':'ACTUAL_CPU_C_CHECKPOINT','evaluation_window_sha256':'fixed',
                    'consumed':32,'updates':4,'actual_elapsed_seconds':10,'at':'2026-10-23T00:00:00+09:00',
                    'validation':{'loss':1.0},'generation':[]}
            metadata={'schema':1,'identity':info['identity'],'config':info['config'],'consumed':32,'updates':4,
                      'sampler':[1,16,3,1,32],'state':{'last_validation':metric['validation'],'last_generation':[]}}
            c.write(run/'run.json',info);c.write(run/'progress.json',{'windows':16});(run/'metrics.jsonl').write_text(json.dumps(metric)+'\n')
            np.savez(run/'latest.npz',metadata=np.frombuffer(json.dumps(metadata).encode(),dtype=np.uint8),
                     parameters=np.zeros(2,dtype=np.float32),moment1=np.zeros(2,dtype=np.float32),
                     moment2=np.zeros(2,dtype=np.float32),visited=np.ones(1,dtype=np.uint8))
            control={'run_dir':str(run),'milestones':[{'epoch':e,'status':'pending'} for e in [2,3,5]]}
            jobs=c.preserve_milestones(control);self.assertEqual([j['status'] for j in jobs],['archived','pending','pending'])
            checkpoint=run/'checkpoint-epoch-2.npz';before=c.identity(checkpoint)
            jobs[0]['status']='pending';self.assertEqual(c.preserve_milestones(control)[0]['status'],'archived');self.assertEqual(c.identity(checkpoint),before)
            side=checkpoint.with_suffix('.json');broken=c.read(side);broken['sha256']='wrong';c.write(side,broken);jobs[0]['status']='pending'
            with self.assertRaisesRegex(ValueError,'mismatch'):c.preserve_milestones(control)
            self.assertEqual(c.identity(checkpoint),before)


if __name__=='__main__':unittest.main()
