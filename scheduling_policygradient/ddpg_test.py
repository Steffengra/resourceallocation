import matplotlib.pyplot as plt

from scheduling_policygradient.ddpg_runner import Runner


def main():
    runner = Runner()
    runner.test()

    plt.show()


if __name__ == '__main__':
    main()
