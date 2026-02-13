# Execution Management System

## Overview

This document describes the execution management improvements implemented to ensure reliable, non-conflicting benchmark runs with proper user feedback and control.

## 1. Unique Execution Directory Naming

### Implementation

**File Modified**: `pipeline/orchestrator.py`

**Previous Behavior**:
- Results stored in `results/<YYYY-MM-DD>/<experiment>/`
- Multiple runs on the same day would overwrite each other

**New Behavior**:
- Results stored in `results/<YYYY-MM-DD_HH-MM-SS>/<experiment>/`
- Each run gets a unique timestamp-based directory

**Code Changes**:
```python
# Line ~1669 in orchestrator.py
timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
out_dir = RESULTS_DIR / timestamp_str / experiment
```

**Timestamp Format Decision**:
- **Format**: `YYYY-MM-DD_HH-MM-SS` (e.g., `2026-02-13_14-30-45`)
- **Timezone**: UTC (consistent across deployments)
- **Precision**: Second-level precision
- **Collision Handling**: 
  - Sufficient for typical use cases (runs separated by >1 second)
  - If millisecond precision is needed, format can be extended to `%Y-%m-%d_%H-%M-%S-%f`
  - Alternatively, could append UUID suffix for absolute uniqueness

**Backward Compatibility**:
- Existing result-reading logic (`ResultsLoader`) scans all subdirectories
- No changes needed to read old date-only directories
- New timestamp directories are automatically discovered
- Sort order naturally chronological (lexicographic sort works correctly)

## 2. Execution Locking + Loading Indicator

### Backend Implementation

**File Modified**: `webapp/backend/app/services/benchmark_runner.py`

**Locking Mechanism**:
- `_active_job: str | None` tracks currently running job
- `is_busy` property prevents concurrent executions
- API endpoint returns HTTP 409 (Conflict) if busy

**Code**:
```python
@property
def is_busy(self) -> bool:
    return self._active_job is not None
```

### Frontend Implementation

**Files Modified**:
- `webapp/frontend/src/pages/RunBenchmarks.tsx`
- `webapp/frontend/src/components/benchmark/BenchmarkForm.tsx`

**State Management**:
```typescript
const [status, setStatus] = useState<string>("idle");
const [jobId, setJobId] = useState<string | null>(null);
const isRunning = status === "running" || status === "starting";
```

**UI States**:
1. **Idle**: Button enabled, no spinner
2. **Starting**: Button disabled, spinner appears
3. **Running**: Button disabled, spinner visible, cancel button shown
4. **Completed**: Success message, auto-redirect after 2s
5. **Failed**: Error message, button re-enabled
6. **Cancelled**: Cancellation message, button re-enabled

**Visual Indicators**:
- **Spinner**: `Loader2` icon with `animate-spin` (from `lucide-react`)
- **Disabled Button**: Opacity reduced, cursor changed to `not-allowed`
- **Status Display**: Shows job ID and current state
- **Cancel Button**: Red button with X icon, appears only when running

**State Flow**:
```
User clicks "Run Benchmark"
  ↓
status = "starting" → Button disabled
  ↓
WebSocket connects
  ↓
status = "running" → Loading indicator + Cancel button shown
  ↓
Execution completes OR user cancels
  ↓
status = "completed" / "failed" / "cancelled" → Button re-enabled
```

## 3. Cancel Ongoing Execution

### Backend Cancellation Mechanism

**File Modified**: `webapp/backend/app/services/benchmark_runner.py`

**Implementation Strategy**: Cooperative process termination

**Code Flow**:
```python
async def cancel(self, job_id: str) -> bool:
    # 1. Verify job is actually running
    if job_id not in self._processes or job_id in self._finished:
        return False
    
    proc = self._processes.get(job_id)
    if proc is None or proc.returncode is not None:
        return False
    
    # 2. Graceful termination (SIGTERM)
    proc.terminate()
    
    # 3. Wait up to 2 seconds for graceful shutdown
    try:
        await asyncio.wait_for(proc.wait(), timeout=2.0)
    except asyncio.TimeoutError:
        # 4. Force kill if process doesn't respond (SIGKILL)
        proc.kill()
        await proc.wait()
    
    # 5. Update status and persist to disk
    status.status = "cancelled"
    status.finished_at = datetime.now(timezone.utc).isoformat()
    self._persist_job(job_id)
    
    # 6. Notify subscribers via WebSocket
    for q in self._subscribers.get(job_id, []):
        q.put_nowait("__CANCELLED__")
        q.put_nowait("__EOF__")
```

**Signal Handling**:
- **SIGTERM** (15): Graceful termination request
  - Process receives signal and can clean up
  - orchestrator.py subprocess terminates cleanly
- **SIGKILL** (9): Force termination (fallback after 2s timeout)
  - Immediate process termination
  - No cleanup, but prevents hung processes

**Partial Results Handling**:
- Results are only written at completion via `write_results()`
- Cancelled processes never reach the write phase
- No partial/corrupt output directories created
- System returns to idle state immediately

**Consistency Guarantees**:
- Job status persisted to disk: `logs/jobs/<job_id>.json`
- Process handle cleaned from `_processes` dict
- `_active_job` set to `None` (unlocks runner)
- Subscribers notified synchronously before returning
- `_finished` set updated to mark job as done

### API Endpoint

**File Modified**: `webapp/backend/app/api/benchmark.py`

**Endpoint**: `POST /api/benchmark/jobs/{job_id}/cancel`

**Response Codes**:
- **200**: Successfully cancelled, returns updated `BenchmarkStatus`
- **404**: Job ID not found
- **409**: Job not in running state (already completed/failed/cancelled)
- **500**: Cancellation failed (process already terminated)

**Example Request**:
```bash
curl -X POST http://localhost:8000/api/benchmark/jobs/abc123def456/cancel
```

**Example Response**:
```json
{
  "job_id": "abc123def456",
  "status": "cancelled",
  "started_at": "2026-02-13T14:30:00Z",
  "finished_at": "2026-02-13T14:30:15Z"
}
```

### Frontend Cancellation

**Files Modified**:
- `webapp/frontend/src/api/client.ts` (API function)
- `webapp/frontend/src/pages/RunBenchmarks.tsx` (UI handler)

**Cancel Button**:
- Only visible when `isRunning === true`
- Styled with red theme to indicate destructive action
- Shows "Cancelling..." text while request is in-flight
- Disabled during cancellation to prevent double-clicks

**WebSocket Handling**:
```typescript
// server sends: {"type": "log", "line": "__CANCELLED__"}
if (msg.line === "__CANCELLED__") {
  onLine("Benchmark execution cancelled by user.");
}
```

**Error Handling**:
```typescript
const handleCancel = useCallback(async () => {
  if (!jobId || isCancelling) return;
  setIsCancelling(true);
  try {
    await cancelJob(jobId);
    setStatus("cancelled");
  } catch (err) {
    console.error("Failed to cancel job:", err);
    setStatus("error");
  } finally {
    setIsCancelling(false);
  }
}, [jobId, isCancelling]);
```

**User Feedback**:
1. Click "Cancel Execution" button
2. Button text changes to "Cancelling..."
3. Backend terminates process
4. WebSocket receives cancellation notification
5. UI shows: "Benchmark execution cancelled."
6. Status badge in job history shows orange "cancelled"
7. Run benchmark button re-enabled

## Logging Behavior

### Cancellation Logs

**Backend Logs** (`logs/<job_id>.log`):
```
Starting benchmark: python3 /path/to/orchestrator.py --experiments all --n 1000
======================================================================
 Experiment: frame_rotation_bpn (N=1000, seed=42)
======================================================================
  Generating inputs...
  Running ERFA adapter (reference)...
Benchmark execution cancelled by user.
```

**Job Metadata** (`logs/jobs/<job_id>.json`):
```json
{
  "job_id": "abc123def456",
  "status": "cancelled",
  "started_at": "2026-02-13T14:30:00Z",
  "finished_at": "2026-02-13T14:30:15Z"
}
```

**Application Logs**:
```
INFO:app.services.benchmark_runner:Starting benchmark: python3 ...
INFO:app.services.benchmark_runner:Cancelling job abc123def456
INFO:app.services.benchmark_runner:Job abc123def456 cancelled successfully
```

## Thread/Process Safety

### Race Condition Prevention

1. **Single Active Job**:
   - `_active_job` lock prevents concurrent `start()` calls
   - API returns 409 if `is_busy` is True

2. **Process Handle Management**:
   - `_processes` dict accessed only in async event loop
   - All modifications are atomic dict operations
   - No threading, only asyncio (single-threaded event loop)

3. **Status Updates**:
   - Status object updates are synchronous
   - Disk persistence happens after in-memory update
   - Finally block guarantees cleanup even on exception

4. **WebSocket Subscribers**:
   - Queue operations are `put_nowait()` (non-blocking)
   - `QueueFull` exceptions are caught and ignored
   - Subscriber cleanup in `unsubscribe()` is idempotent

### Deterministic Logging

1. **Log File Creation**:
   - Each job gets unique file: `logs/<job_id>.log`
   - File opened in `_read_output()` with exclusive write

2. **Log Line Buffering**:
   - Lines stored in `_log_buffers[job_id]`
   - Persisted to disk immediately (flush after each line)
   - Subscribers receive lines in order via queue

3. **Job Metadata Persistence**:
   - `_persist_job()` called at key lifecycle points:
     - Job start (initial status)
     - Job completion (final status)
     - Job cancellation (cancelled status)

4. **Restoration on Restart**:
   - `_restore_jobs()` scans `logs/jobs/*.json`
   - Reconstructs job history from disk
   - Log buffers reloaded from log files

## Testing Checklist

### Manual Testing Steps

1. **Normal Execution**:
   - [ ] Start benchmark with 1 experiment, N=100
   - [ ] Verify loading spinner appears
   - [ ] Verify "Run Benchmark" button is disabled
   - [ ] Wait for completion
   - [ ] Verify success message and redirect
   - [ ] Check results directory has timestamp format: `YYYY-MM-DD_HH-MM-SS`

2. **Concurrent Prevention**:
   - [ ] Start benchmark
   - [ ] Try to start another (should see error or disabled state)
   - [ ] Verify HTTP 409 error in browser console
   - [ ] Wait for first to complete
   - [ ] Verify can start new benchmark

3. **Cancellation (Quick)**:
   - [ ] Start benchmark with N=10000
   - [ ] Click "Cancel Execution" immediately
   - [ ] Verify "Cancelling..." text appears
   - [ ] Verify process terminates within 2 seconds
   - [ ] Verify status shows "cancelled"
   - [ ] Verify no results directory created
   - [ ] Verify "Run Benchmark" button re-enabled

4. **Cancellation (Mid-Run)**:
   - [ ] Start benchmark with all experiments, N=1000
   - [ ] Wait for 1-2 experiments to complete
   - [ ] Click "Cancel Execution"
   - [ ] Verify cancellation message in logs
   - [ ] Verify partial results NOT written (or isolated)
   - [ ] Verify job history shows "cancelled"

5. **Multiple Runs Same Minute**:
   - [ ] Run benchmark with N=100 (completes quickly)
   - [ ] Immediately run another
   - [ ] Verify two distinct directories created
   - [ ] Verify directories have different timestamps

6. **Job History**:
   - [ ] Complete a benchmark successfully
   - [ ] Cancel a benchmark mid-run
   - [ ] Fail a benchmark (e.g., invalid parameters)
   - [ ] Verify job history shows all three with correct status colors:
     - ✅ Green for completed
     - ❌ Red for failed  
     - 🟠 Orange for cancelled
   - [ ] Click "View Logs" for each, verify logs display

7. **WebSocket Reconnection**:
   - [ ] Start benchmark
   - [ ] Open browser DevTools Network tab
   - [ ] Block WebSocket connection or refresh page mid-run
   - [ ] Verify UI handles disconnection gracefully
   - [ ] Verify job still completes in background
   - [ ] Verify job history updates when page refreshed

## Known Limitations

1. **Timestamp Precision**:
   - Second-level precision may cause collisions if:
     - Multiple benchmarks triggered programmatically <1s apart
     - System clock adjusted during execution
   - **Mitigation**: Add milliseconds (`%f`) or UUID suffix if needed

2. **Orphaned Processes**:
   - If FastAPI server crashes during execution:
     - orchestrator.py subprocess continues running
     - No mechanism to reattach on server restart
   - **Mitigation**: Consider process group management or systemd integration

3. **Disk Space**:
   - Each run creates a new directory
   - No automatic cleanup of old results
   - **Mitigation**: Implement result archival/deletion in UI

4. **Cancellation Timing**:
   - 2-second timeout before SIGKILL may be too short for large experiments
   - **Mitigation**: Make timeout configurable based on experiment type

## Future Enhancements

1. **Progress Reporting**:
   - Parse orchestrator output for experiment completion
   - Show progress bar (e.g., "3/7 experiments complete")

2. **Result Retention Policy**:
   - Auto-delete results older than N days
   - Mark runs for preservation

3. **Parallel Execution**:
   - Allow multiple concurrent runs (with resource limits)
   - Queue system for pending runs

4. **Pause/Resume**:
   - Checkpoint experiment state
   - Resume from last completed experiment

5. **Priority Queue**:
   - High-priority runs can preempt low-priority ones

6. **Email/Webhook Notifications**:
   - Notify on completion/failure for long-running jobs
