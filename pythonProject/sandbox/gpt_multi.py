import multiprocessing as mp
import random


class Peel:

    def __init__(self, manager):
        self.spots = manager.Value('b', True)

class Banana:


    def __init__(self, manager):
        self.is_bad = manager.Value('b', random.choice([True, False]))  # 'b' for bool
        self.discard = manager.Value('b', False)
        self.sorted = manager.Value('b', False)
        self.peel = Peel(manager)
        self.peels = []

def sort_bananas(lock, banana):
    with lock:
        if banana.is_bad.value:
            banana.discard.value = True
        banana.sorted.value = True
        banana.peel.spots.value = False
        banana.peels.append()

def main():
    num_banana = 100
    manager = mp.Manager()
    lock = manager.Lock()
    bananas = [Banana(manager) for _ in range(num_banana)]

    with mp.Pool(processes=2) as pool:
        # Pass the whole banana object now
        pool.starmap(sort_bananas, [(lock, banana) for banana in bananas])

    print(sum([banana.sorted.value for banana in bananas]))
    print(sum([banana.discard.value for banana in bananas]))
    print(sum([banana.peels[0].spots.value for banana in bananas]))

if __name__ == '__main__':
    mp.freeze_support()
    main()