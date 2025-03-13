"""
    SMC.CloudPickleTest.py
"""
from time import sleep
import multiprocessing
import cloudpickle
from concurrent.futures import ProcessPoolExecutor, wait
from pymc.util import CustomProgress
from rich.progress import (
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

class Problematic:
    def __init__(self):
        pass
    def _method(self):
        pass

def exec_func(
    chain,
    progress_dict,
    task_id,
):
    sleep(3)
    problem = Problematic()
    results = (
        chain,
        problem,
    )
    return cloudpickle.dumps(results)

def run_processes():
    with CustomProgress(
        TextColumn("{task.description}"),
        SpinnerColumn(),
        TimeRemainingColumn(),
        TextColumn("/"),
        TimeElapsedColumn(),
        TextColumn("{task.fields[status]}"),
        disable=False,
    ) as progress:
        kwargs = {'a':1, 'b':2}
        workers = 4
        chains = 2
        futures = []  # keep track of the jobs
        with multiprocessing.Manager() as manager:
            # this is the key - we share some state between our
            # main process and our worker functions
            _progress = manager.dict()

            with ProcessPoolExecutor(max_workers=workers) as executor:
                for c in range(chains):  # iterate over the jobs we need to run
                    # set visible false so we don't have a lot of bars all at once:
                    task_id = progress.add_task(f"Chain {c}", status="Stage: 0 Beta: 0")
                    futures.append(
                        executor.submit(
                            exec_func,
                            c,
                            _progress,
                            task_id,
                        )
                    )

                # monitor the progress:
                done = []
                remaining = futures
                while len(remaining) > 0:
                    finished, remaining = wait(remaining, timeout=0.1)
                    done.extend(finished)
                    for task_id, update_data in _progress.items():
                        stage = update_data["stage"]
                        beta = update_data["beta"]
                        # update the progress bar for this task:
                        progress.update(
                            status=f"Stage: {stage} Beta: {beta:.3f}",
                            task_id=task_id,
                            refresh=True,
                        )

    return tuple(cloudpickle.loads(r.result()) for r in done)

if __name__ == '__main__':
    run_processes()