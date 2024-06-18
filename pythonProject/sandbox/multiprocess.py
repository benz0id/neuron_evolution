import multiprocessing as mp
import random
from typing import Union


def sort_bananas(bananas, banana_ind):
    if bananas[banana_ind].is_bad:
        bananas[banana_ind].discard = True
    bananas[banana_ind].sorted = True

class Banana:
    is_bad: bool
    discard: Union[bool, None]
    sorted: bool

    def __init__(self):
        self.discard = None
        self.sorted = False
        self.is_bad = random.choice([True, False])

def main():


    num_banana = 100

    manager = mp.Manager()

    bananas = manager.list()
    for _ in range(num_banana):
        bananas.append(Banana())

    with mp.Pool(processes=2) as pool:
        # Map process_node function to each node
        pool.starmap(sort_bananas, [(bananas, banana_ind)
                                    for banana_ind in range(len(bananas))])

    print(sum([banana.sorted for banana in bananas]))

if __name__ == '__main__':
    mp.freeze_support()
    main()
