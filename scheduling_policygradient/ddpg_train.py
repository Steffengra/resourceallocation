
import matplotlib.pyplot as plt

from scheduling_policygradient.ddpg_runner import Runner

import sys
sys.path.append(join('/Py', 'robbot'))
from robbot_imports.SendMessage import send


def main():
    runner = Runner()
    runner.train()

    send('Simulation complete')
    plt.show()


if __name__ == '__main__':
    main()
