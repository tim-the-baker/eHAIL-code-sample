import numpy as np
import matplotlib.pyplot as plt

def save_load_data():
    filename = ""
    save, load = None, None

    if load:
        print(f"Attempting to load {filename}.")
        try:
            Z_hats, Z_stars = np.load(filename)
            loaded = True
            print("Data loaded successfully.")
        except FileNotFoundError:
            print("Data file not found. Simulating results now.")
            loaded = False
            Z_hats, Z_stars = None, None  # TODO: replace with experiment function
    else:
        print("Load set to False. Simulating results now.")
        loaded = False
        Z_hats, Z_stars = None, None  # TODO: replace with experiment function


def single_line_plot(x, y):
    fig1 = plt.figure(figsize=(3.5, 2), dpi=300)
    ax = fig1.gca()
    fig1.tight_layout()
    ax.plot(x, y)

    fig2 = plt.figure(figsize=(2,3.5), dpi=300)
    ax2 = fig2.gca()
    ax2.plot(x, -y, color='red')
    fig2.tight_layout()

    fig3 = plt.figure(figsize=(3.5, 2), dpi=300)
    ax3 = fig3.add_subplot(221)
    ax4 = fig3.add_subplot(222)
    ax5 = fig3.add_subplot(212)
    ax3.plot(x, y, color='green')
    ax4.plot(x, y, color='purple')
    ax5.plot(x, y, color='orange')
    fig3.tight_layout()
    plt.show(block=True)


def grid_of_plots():
    pass


if __name__ == '__main__':
    x = np.linspace(0, 2*np.pi, 1000)
    y = np.sin(x)
    single_line_plot(x, y)