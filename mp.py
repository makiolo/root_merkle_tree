from multiprocessing.pool import Pool, ThreadPool
from functools import partial


def iterate_process_pool(is_mp, functor, iterable, num_process, maxtasksperchild=None):
    if is_mp:
        p = Pool(num_process, maxtasksperchild=maxtasksperchild)
    else:
        p = ThreadPool(num_process)
    try:
        return p.map(functor, iterable)
        # for result in p.imap_unordered(functor, iterable):
        #     yield result
    finally:
        p.close()


def sequence_chunked(generator, split_each):
    i = 0
    chunk = []
    for elem in generator():
        chunk.append(elem)
        if len(chunk) >= split_each:
            yield chunk, i
            i += 1
            chunk = []
    if len(chunk) > 0:
        yield chunk, i


def chunks_generator_superchunked(generator, split_each, num_process):
    superchunk = []
    for elem in sequence_chunked(generator, split_each):
        superchunk.append(elem)
        if len(superchunk) >= num_process:
            yield superchunk
            superchunk = []
    if len(superchunk) > 0:
        yield superchunk


def process_parallel(is_mp, generator, processor, chunk_each, num_process):
    '''
    :param is_mp:
    :param generator: generator function
    :param processor: processor function
    :param chunk_each: chunk size
    :param num_process: number process in parallel
    :return:
    '''
    for chunk in chunks_generator_superchunked(generator, chunk_each, num_process):
        for result in iterate_process_pool(is_mp, processor, iter(chunk), num_process):
            yield result


def process_pipeline(parallel, is_mp, generator, processor, chunk_each, num_process, context):

    generator_plus = partial(generator, context)
    processor_plus = partial(processor, context)
    if parallel:
        for result in process_parallel(is_mp, generator_plus, processor_plus, chunk_each, num_process):
            yield result
    else:
        for chunk in sequence_chunked(generator_plus, chunk_each):
            result = processor_plus(chunk)
            yield result
from multiprocessing.pool import Pool, ThreadPool
from functools import partial


def iterate_process_pool(is_mp, functor, iterable, num_process, maxtasksperchild=None):
    if is_mp:
        p = Pool(num_process, maxtasksperchild=maxtasksperchild)
    else:
        p = ThreadPool(num_process)
    try:
        return p.map(functor, iterable)
        # for result in p.imap_unordered(functor, iterable):
        #     yield result
    finally:
        p.close()


def sequence_chunked(generator, split_each):
    i = 0
    chunk = []
    for elem in generator():
        chunk.append(elem)
        if len(chunk) >= split_each:
            yield chunk, i
            i += 1
            chunk = []
    if len(chunk) > 0:
        yield chunk, i


def chunks_generator_superchunked(generator, split_each, num_process):
    superchunk = []
    for elem in sequence_chunked(generator, split_each):
        superchunk.append(elem)
        if len(superchunk) >= num_process:
            yield superchunk
            superchunk = []
    if len(superchunk) > 0:
        yield superchunk


def process_parallel(is_mp, generator, processor, chunk_each, num_process):
    '''
    :param is_mp:
    :param generator: generator function
    :param processor: processor function
    :param chunk_each: chunk size
    :param num_process: number process in parallel
    :return:
    '''
    for chunk in chunks_generator_superchunked(generator, chunk_each, num_process):
        for result in iterate_process_pool(is_mp, processor, iter(chunk), num_process):
            yield result


def process_pipeline(parallel, is_mp, generator, processor, chunk_each, num_process, context):

    generator_plus = partial(generator, context)
    processor_plus = partial(processor, context)
    if parallel:
        for result in process_parallel(is_mp, generator_plus, processor_plus, chunk_each, num_process):
            yield result
    else:
        for chunk in sequence_chunked(generator_plus, chunk_each):
            result = processor_plus(chunk)
            yield result


### BEGIN TEST

def toy_generator(context):
    for i in range(1000):
        yield i


def toy_process(context, chunkset):
    print('context = {}'.format(context))
    print('chunk = {}'.format(chunkset[0]))
    time.sleep(1)
    return chunkset[0]

### END TEST


if __name__ == '__main__':

    begin = time.time()
    context = {
        'parm1': 'parm1',
        'parm2': 'parm2',
    }
    for result in process_pipeline(True, True, toy_generator, toy_process, 100, 10, context):
        print(result)
    print('test parallel, elapsed: {}'.format(time.time() - begin))


            
