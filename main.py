from simple_phase_nn.models import *
from simple_phase_nn.utils import *
from simple_phase_nn.datasets import *
from simple_phase_nn.plots import *

from simple_phase_nn.train import train


if __name__ == "__main__":
    
    cf = AttrDict() # Describes all the 'knobs' of an experiment
    cf.seeds = [42, 43, 44, 45, 46]  # List of seeds for reproducibility

    # TRAINING
    cf.epochs = 200
    cf.learning_rate = 5e-5
    cf.weight_decay = 1e-5        # L2 regularization on weights
    cf.patience = 10              # early stopping patience, in epochs
    cf.init = "kaiming"           # weight initialization method, only "kaiming" is implemented so far

    # DATASET
    cf.dataset = "XXZ"
    cf.label_param = "Jz"      # label parameter for the Paris datasets; 't', 'delta', 'omega' for Ising, 't' for XY, 'beta' for ILGT
    cf.normalize_labels = True    # normalize labels to [0, 1] range, only for regression
    cf.kernels = None

    for seed in cf.seeds:
        print(f"\nRunning experiment with dataset={cf.dataset}, seed={seed}\n")
        cf.seed = seed
        cf.logdir = f"logs/{cf.dataset}/seed_{cf.seed}"

        train(cf)

        metrics = load_json(cf.logdir, "metrics.json")
        plot_history(metrics, cf, "history.png")
        plot_phase_transition(metrics, cf, "phase_transition.png")