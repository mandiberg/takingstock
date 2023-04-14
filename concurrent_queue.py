import time
from functools import wraps


def timeit(method):
    @wraps(method)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = method(*args, **kwargs)
        end_time = time.time()
        print(f"{method.__name__} => {(end_time-start_time)*1000} ms")

        return result

    return wrapper


import hashlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


def hash_one(n):
    """A somewhat CPU-intensive task."""

    for i in range(1, n):
        hashlib.pbkdf2_hmac("sha256", b"password", b"salt", i * 10000)

    return "done"


@timeit
def hash_all(n):
    """Function that does hashing in serial."""

    with ProcessPoolExecutor(max_workers=10) as executor:
        for arg, res in zip(range(n), executor.map(hash_one, range(n), chunksize=2)):
            pass

    return "done"


if __name__ == "__main__":
    hash_all(20)


# import concurrent.futures
# import time
 
# def cube(x):
#     return x**3
 
# if __name__ == "__main__":
#     with concurrent.futures.ProcessPoolExecutor(3) as executor:
#         start_time = time.perf_counter()
#         result = list(executor.map(cube, range(1,1000)))
#         finish_time = time.perf_counter()
#     print(f"Program finished in {finish_time-start_time} seconds")
#     print(result)