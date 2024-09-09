import subprocess

from pydantic import BaseModel
from functools import wraps
import click
import time


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        time_unit="seconds"
        start_time=time.perf_counter()
        result = func(*args, **kwargs)
        end_time=time.perf_counter()
        total_time=end_time-start_time
        if total_time>60:
            total_time = total_time / 60
            time_unit='minutes'
        print(f'Function {func.__name__} took {total_time:.4f} {time_unit}')
        return result
    return timeit_wrapper



def run_script(script_name):
    subprocess.run(['python', f'ml_pipeline/{script_name}'], check=True)


