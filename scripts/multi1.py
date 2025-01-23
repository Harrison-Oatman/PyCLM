import numpy as np
import multiprocessing
from time import time, sleep
from abc import ABC, abstractmethod

# a is controller
# b is child: requires arrays from a and provides arrays to a
# c is child: requires arrays from a and provides arrays to a

k = 3200


class Flag(ABC):
    msg = "BASE_FLAG"


class ArrayReady(Flag):
    msg = "array_ready"


class CloseConnection(Flag):
    msg = "close"


def process_a(data_queue, msg_queue):

    def handle_array_b(arr):
        print(f"array received")
        print(arr.shape)

    def handle_message_b(msg):
        if msg.msg == "close":
            return 1

        elif msg.msg == "array_ready":
            tstart = time()
            arr = data_queue.get()
            print(f"took: {time() - tstart}")

            handle_array_b(arr)

        else:
            raise NotImplementedError(f"Unknown message {msg.msg}")

        return 0

    while True:
        if not msg_queue.empty():

            close = handle_message_b(msg_queue.get())

            if close:
                return


def process_b(data_queue, msg_queue):

    for j in range(10):
        random_array = np.random.random((k, k))

        # tstart = time()
        data_queue.put(random_array)
        # print(f"took: {time() - tstart}")

        msg_queue.put(ArrayReady)

        sleep(0.5)

    msg_queue.put(CloseConnection)


if __name__ == "__main__":
    #
    # a_to_b, b_to_a = multiprocessing.Pipe()
    # a_to_b_msg, b_to_a_msg = multiprocessing.Pipe()
    #
    # pa = multiprocessing.Process(target=process_a, args=(a_to_b, a_to_b_msg))
    # pb = multiprocessing.Process(target=process_b, args=(b_to_a, b_to_a_msg))

    ab_msg_queue = multiprocessing.Queue()
    ab_data_queue = multiprocessing.Queue()

    pa = multiprocessing.Process(target=process_a, args=(ab_data_queue, ab_msg_queue))
    pb = multiprocessing.Process(target=process_b, args=(ab_data_queue, ab_msg_queue))

    pa.start()
    pb.start()

    pa.join()
    pb.join()



