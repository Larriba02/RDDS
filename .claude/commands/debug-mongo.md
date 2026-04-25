---
description: Hand off a MongoDB / Atlas error to the mongo-debugger subagent.
allowed-tools: Bash, Read, Grep, Glob, Task
argument-hint: "<paste the error message after the slash command>"
---

# /debug-mongo

Wrap the `mongo-debugger` subagent for a specific Atlas / pymongo problem.

## Procedure

1. Read the error the user pasted as `$ARGUMENTS`. If empty, ask the user to
   paste the full traceback (not a paraphrase).

2. Gather quick context, *without* hitting the database:

   ```bash
   python -c "from pymongo import __version__ as v; print('pymongo', v)"
   git log -n 5 --oneline -- src/db/
   ```

   Also `cat .env.example` (never `.env`).

3. Invoke the `mongo-debugger` subagent with the Task tool. Pass it:
   - The full error.
   - The output of step 2.
   - The Atlas reference doc: `DOCUMENTATION/IN DETAIL/mongo.md` §7
     (troubleshooting matrix).

4. Relay the subagent's diagnosis back to the user. Apply fixes only after
   the user approves.

## Common failure shapes (route directly without escalation)

- `OperationFailure: bad auth` → password is the **DB user** password, not the
  Atlas web account password. See `mongo.md` §2.
- `ConfigurationError: SRV resolution failed` → DNS/VPN issue; suggest the
  non-SRV connection string from Atlas → Connect → Standard.
- `ServerSelectionTimeoutError` → Atlas Network Access list does not include
  the user's IP. Add it (or `0.0.0.0/0`) and retry.

## Do not

- Do not log or print `MONGO_URI` or any password. Quote tracebacks redacted
  if needed.
- Do not write `MONGO_URI` to `.env` based on what the user pastes in chat.
  Tell them to put it in `.env` themselves.
