---
description: Run the MongoDB Step 1 smoke test and interpret the result.
allowed-tools: Bash, Read
argument-hint: ""
---

# /smoke-test

Run the two Step 1 verification scripts and report the outcome.

## Procedure

1. Confirm the venv is active (the scripts depend on `pymongo` and
   `python-dotenv`). If not, instruct the user to activate it first and stop —
   do not try to install dependencies.
2. Run, in this order, from the repo root:

   ```bash
   python -m src.db.setup_atlas
   python -m src.db.test_connection
   ```

3. Capture the output of each command verbatim.

## Reporting

- **All `OK`:** report success, name the three collections that were verified
  (`images_metadata`, `experiments`, `predictions`), and confirm Step 1 is green
  on this machine.
- **`setup_atlas` fails:** the cluster, user, or network is misconfigured.
  Suggest invoking the `mongo-debugger` subagent with the exact error.
- **`setup_atlas` succeeds but `test_connection` fails:** indexes were created
  but writes are blocked. Almost always a role / permission issue on the Atlas
  user. Quote the error and route to `mongo-debugger`.
- **`ModuleNotFoundError: No module named 'src'`:** the user ran the script as
  a file instead of a module. Tell them to run with `python -m src.db.…` from
  the repo root.

## Do not

- Do not modify `.env`. If `MONGO_URI` is missing or empty, tell the user to
  fix it themselves and re-run.
- Do not retry repeatedly. If both commands fail twice in a row with the same
  error, hand off to `mongo-debugger`.
