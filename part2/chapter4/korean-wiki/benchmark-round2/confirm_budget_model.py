"""Confirm both budget-model observations with a shared longer fixed workload.

The v1 128-update observations were 5-7 seconds. Retain them as a pilot and
double work for both architectures; never tune a workload per candidate.
"""
from pathlib import Path
import json
import run_budget_model as campaign

campaign.OUTPUT = Path(__file__).resolve().parent.parent / 'reports/nplm-budget-model-20261018-v2.json'
campaign.STEPS = 256
campaign.EVENTS = campaign.BATCH * campaign.STEPS

if __name__ == '__main__':
    wrapper = Path(__file__).resolve()
    before = campaign.identity(wrapper)
    code = campaign.main()
    after = campaign.identity(wrapper)
    assert before == after
    report = json.loads(campaign.OUTPUT.read_text())
    report['confirmation_wrapper'] = {'path': str(wrapper), 'before': before, 'after': after}
    report['confirmation_policy'] = 'Preserved v1 short pilot; common work doubled to 256 updates / 131072 targets for both architectures; no candidate-specific retuning.'
    campaign.atomic_json(campaign.OUTPUT, report)
    raise SystemExit(code)
