import subprocess

#process_train = subprocess.run()
#process_test = subprocess.run()

from pydantic import BaseModel
from functools import wraps
import click
import time

class ModelToExport(BaseModel):
    modelName: str
    version: str

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

@click.command()
@click.option('-env', '--environnement', default='DEV', help='e.g DEV TEST PROD')
@click.option('--ml_model_names', default='ml_model_01.json', help='models list to export')
@timeit
def something(environnement, ml_model_names):
    print(environnement)   

# appeler le retrain avec une cli sur le nom/id du modele
if __name__=="__main__":
    something()