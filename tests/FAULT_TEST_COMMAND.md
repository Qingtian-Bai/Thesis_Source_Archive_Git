# Fault-injection test command

Run from the frozen-release root:

```powershell
powershell -ExecutionPolicy Bypass -File tests\run_fault_tests.ps1
```

To use another compatible Python environment:

```powershell
powershell -ExecutionPolicy Bypass -File tests\run_fault_tests.ps1 -PythonExecutable "C:\path\to\python.exe"
```

The test suite uses synthetic frames, fake capture devices, mocked model
objects, and temporary directories. It does not load participant recordings or
run detector inference. A new timestamped result directory is written under
`runtime_outputs/fault_test_runs/` on each execution. The frozen evidence used
in the dissertation remains under `tests/evidence/fault_tolerance/`.
