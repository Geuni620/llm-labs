"""One explicitly approved restart, preserving the first attempt and deadlines."""
import continue_run as app

app.TAG='resume-20261023-r2'
app.CONTROL=app.WIKI/f'reports/{app.TAG}-control.json'
app.PLAN=app.WIKI/'resume-run-20261023-r2.execution.json'
app.SOURCE=app.WIKI/'training-runs/resume-cpu-20261023-H128-B512-to5'
app.SOURCE_RESULT=app.WIKI/'reports/resume-20261023-result.json'
app.RUN=app.WIKI/'training-runs/resume-cpu-20261023-r2-H128-B512-to5'
app.PREFLIGHT=app.WIKI/'training-runs/resume-preflight-20261023-r2-H128-B512'
source_plan=app.read(app.SOURCE/'run-plan.json')
app.FIXED_DEADLINES={key:source_plan[key] for key in ['stop_training_by','exit_by']}
app.RESTART_APPROVAL={'approved':True,'user_response':'같은 설정으로 재개',
                     'scope':'Resume the saved 1.817666644 epoch state to cumulative 2/3/5 with identical CPU4/model/optimizer/guard and unchanged inherited deadlines. This is not permission for automatic future restarts.'}

if __name__=='__main__':app.main()
