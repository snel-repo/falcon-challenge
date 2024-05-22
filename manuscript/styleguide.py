import matplotlib.pyplot as plt
import seaborn as sns

def set_style():
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 7,
        'axes.titlesize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'pdf.use14corefonts': True
    })