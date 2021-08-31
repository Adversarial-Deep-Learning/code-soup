import torch


class Checkpoints:
    """
    A class to save and load checkpoints
    """

    @classmethod
    def save(self, PATH, model, optimizer, EPOCH=None, LOSS=None):
        """
        Parameters
        ----------
        PATH : str
            - The path where the model is saved
        model:
            - The model which is saved
        optimizer : torch.optim
            - Default: None , optimizer saved at the checkpoint
        EPOCH (optional): int
            - Default:None , epoch
        LOSS (optional):
            - Default: None , loss saved at the checkpoint

        Saves the model state and checkpoint state with optimizer and loss if specified
        """
        checkpoint = {
            "model": model,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": EPOCH,
            "loss": LOSS,
        }
        torch.save(checkpoint, PATH)

    @classmethod
    def load(self, PATH):
        """
        Paramters
        ---------
        PATH: str
            - The path from where the model is loaded

        Returns the loaded model
        """
        checkpoint = torch.load(PATH)
        model = checkpoint["model"]
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model
