from datetime import datetime


log={
    "dataset_id": "",
    "timestamp": str(datetime.today()), 
    "status": "",
    "details": {}
    }



def data_logger(f):
    def wrapper(*args, **kwargs):
        try:
            r = f(*args, **kwargs)

            data = kwargs.get('full_data')
            data_id = kwargs.get('data_id')

            log['status'] = "success"
            log['dataset_id'] = id
            log['details']['nb_rows'] = len(data)
            if r:
                return r
        except Exception as e:
            pass
        return f(*args, **kwargs)
    return wrapper


def ml_logger(f):
    """Train and test logger"""
    pass


def api_loggerf(f):
    pass