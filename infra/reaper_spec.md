# Reaper (separate cron)

Independent of the agent and the gpu wrapper. Every ~5 minutes:
1. List active jobs + start times + budget class from the ledger.
2. Kill any job past its budget-class wall clock or the global spend cap.
3. Update the ledger entry (status=reaped, actual cost), notify on kill.
Runs with its own credentials. Never importable or callable by the agent.
