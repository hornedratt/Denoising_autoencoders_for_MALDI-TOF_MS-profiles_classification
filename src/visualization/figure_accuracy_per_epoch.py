import matplotlib.pyplot as plt
from typing import List
def losses_plot(train_losses: List[float],
                valid_losses: List[float],
                output_path: str) -> None:
    plt.plot(train_losses, label="train_loss")
    plt.plot(valid_losses, label="val_loss")
    plt.legend(loc='best')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.savefig(output_path)
    return None

if __name__ == "__main__":
    losses_plot()