# TODO: know order in queue (burger king style)

import multiprocessing
import threading
import time
from Queue import Empty
from multiprocessing.pool import Pool, ThreadPool
from functools import partial


def iterate_process_pool(is_mp, functor, iterable, num_process, maxtasksperchild=None):
    if is_mp:
        p = Pool(num_process, maxtasksperchild=maxtasksperchild)
    else:
        p = ThreadPool(num_process)
    try:
        # data = [(filter(None, x[0]), x[1]) for x in iterable]
        # data = [x for x in data if len(x[0]) > 0]
        return p.map(functor, iterable)
        # data = list(iterable)
        # usefuls = 0
        # for chunkset in data:
        #     commands = chunkset[0]
        #     for tpl in commands:
        #         command = tpl[0]
        #         if command is not None:
        #             usefuls += 1
        # if usefuls > 1:
        #     return p.map(functor, data)
        # elif usefuls == 1:
        #     return [functor(data[0])]
        # else:
        #     return []
    finally:
        p.close()


def sequence_chunked(generator_plus, split_each):
    i = 0
    chunk = []
    for elem in generator_plus():
        chunk.append(elem)
        if len(chunk) >= split_each:
            yield chunk, i
            i += 1
            chunk = []
    if len(chunk) > 0:
        yield chunk, i


def chunks_generator_superchunked(generator_plus, split_each, num_process):
    superchunk = []
    for elem in sequence_chunked(generator_plus, split_each):
        superchunk.append(elem)
        if len(superchunk) >= num_process:
            yield superchunk
            superchunk = []
    if len(superchunk) > 0:
        yield superchunk


def process_parallel(is_mp, generator_plus, processor_plus, chunk_each, num_process):
    '''
    :param is_mp:
    :param generator: generator function
    :param processor: processor function
    :param chunk_each: chunk size
    :param num_process: number process in parallel
    :return:
    '''
    for chunk in chunks_generator_superchunked(generator_plus, chunk_each, num_process):
        for result in iterate_process_pool(is_mp, processor_plus, iter(chunk), num_process):
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


#################


def generator(context):
    requests = context.get('requests')
    ticket_start = context.get('ticket_start')
    ticket = 0
    exit = False
    while not exit:
        try:
            args = requests.get()
            yield args, ticket
            ticket_start.put_nowait(ticket)
            requests.task_done()
            ticket += 1
        except EOFError:
            exit = True


def processor(context, chunkset):
    lambda_function = context.get('lambda_function')
    commands = chunkset[0]
    rets = []
    for data in commands:
        args = data[0]
        ticket = data[1]
        result = lambda_function(args)
        rets.append((ticket, result))
    return rets


def process_commands(context):
    chunk_each = context.get('chunk_each')
    num_processors = context.get('num_processors')
    results_queue = context.get('results')
    ticket_end = context.get('ticket_end')
    is_mp = context.get('is_mp')
    parallel = context.get('parallel')
    for results in process_pipeline(parallel, is_mp, generator, processor, chunk_each, num_processors, context):
        for ticket, result in results:
            results_queue.put_nowait((ticket, result))
            ticket_end.put_nowait(ticket)


class QueueSystem:
    def __init__(self, lambda_function, parallel, is_mp, chunk_each, num_processors):
        self.m = multiprocessing.Manager()
        # queue events
        self.requests = self.m.Queue()
        self.results = self.m.Queue()
        self.ticket_start = self.m.Queue()
        self.ticket_end = self.m.Queue()
        self.context = {
            'lambda_function': lambda_function,
            'requests': self.requests,
            'results': self.results,
            'ticket_start': self.ticket_start,
            'ticket_end': self.ticket_end,
            'chunk_each': chunk_each,
            'num_processors': num_processors,
            'parallel': parallel,
            'is_mp': is_mp,
        }
        self.th = threading.Thread(target=process_commands, args=(self.context,))
        self.th.setDaemon(True)
        self.th.start()
        self.count = 0
        self.finished = {}
        self.finished_results = {}

    def send(self, command):
        self.requests.put_nowait(command)
        ticket_id = self.ticket_start.get()
        try:
            self.count += 1
            self.finished[ticket_id] = False
        finally:
            self.ticket_start.task_done()
        return ticket_id

    def is_ready(self, ticket):
        stop = False
        while not stop:
            try:
                ticket_id = self.ticket_end.get(timeout=0.01)
                try:
                    self.finished[ticket_id] = True
                    if ticket_id == ticket:
                        break
                finally:
                    self.ticket_end.task_done()
            except Empty:
                stop = True
        return self.finished[ticket]

    def get(self, ticket):
        while self.count > 0:
            ticket_id, results = self.pop()
            if ticket_id == ticket:
                break
        return self.finished_results[ticket]

    def pop(self):
        ticket_id, results = self.results.get()
        try:
            self.count -= 1
            self.finished[ticket_id] = True
            self.finished_results[ticket_id] = results
        finally:
            self.results.task_done()
        return ticket_id, results


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

