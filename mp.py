import os
import time
from multiprocessing.pool import Pool


def data_generator(parm1, parm2):
    for elem in range(1000):
        yield elem


def process_element(picklable):
    chunk, args, kwargs = picklable
    print('chunk = {}'.format(chunk))
    print('args = {}'.format(args))
    print('kwargs = {}'.format(kwargs))
    print(os.getpid())
    return chunk


def _process_parallel(functor, iterable, num_process, chunksize):
    p = Pool(num_process, maxtasksperchild=chunksize)
    try:
        return p.map(functor, iterable, chunksize)
    finally:
        p.close()


def _data_generator_chunked(generator, split_each, *args, **kwargs):
    chunk = []
    for elem in generator(*args, **kwargs):
        chunk.append(elem)
        if len(chunk) >= split_each:
            yield chunk, args, kwargs
            chunk = []
    if len(chunk) > 0:
        yield chunk


def chunk_list(list_tests, n):
    for i in range(0, len(list_tests), n):
        yield list_tests[i:i + n]


def process_parallel(generator, processor, chunk_each, num_process, *args, **kwargs):
    '''
    :param generator: generator function
    :param processor: processor function
    :param chunk_each: chunk size
    :param num_process: process in parallel
    :param args: generator extra args
    :param kwargs: generator extra kwargs
    :return:
    '''
    results = []
    for chunks in chunk_list(list(_data_generator_chunked(generator, chunk_each, *args, **kwargs)), num_process):
        result = _process_parallel(processor, chunks, num_process, 1)
        results.extend(result)
    return [item for sublist in results for item in sublist]


if __name__ == '__main__':

    begin = time.time()
    print(process_parallel(data_generator, process_element, 100, 16, 'parm1', 'parm2'))
    print('test parallel, elapsed: {}'.format(time.time() - begin))
