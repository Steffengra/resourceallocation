import numpy as np
import matplotlib.pyplot as plt

from scheduling.scheduling_config import Config
from scheduling.dqn_runner import Runner


def main():
    runner = Runner()
    runner.test()


if __name__ == '__main__':
    main()