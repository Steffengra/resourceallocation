
import matplotlib.pyplot as plt
from scheduling.dqn_runner import Runner


def main():
    runner = Runner()
    runner.test()

    plt.show()


if __name__ == '__main__':
    main()
