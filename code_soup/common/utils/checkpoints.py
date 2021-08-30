import torch


class Checkpoints:
    """
    A class to save and load checkpoints
    """

    @classmethod
    def save(self, PATH, model, EPOCH, optimizer=None, LOSS=None):
        """
        Parameters
        ----------
        PATH : str
            - The path where the model is saved
        model:
            - The model which is saved
        EPOCH: int
            - Checkpoint
        optimizer (optional):
            - Default: None , optimizer saved at the checkpoint
        LOSS (optional):
            - Default: None , loss saved at the checkpoint

        Saves the model state and checkpoint state with optimizer and loss if specified
        """
        torch.save(
            {
                "epoch": EPOCH,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": LOSS,
            },
            PATH,
        )

    @classmethod
    def load(self, PATH):
        """
        Paramters
        ---------
        PATH: str
            - The path from where the model is loaded

        Returns the loaded model
        """
        return torch.load(PATH)
