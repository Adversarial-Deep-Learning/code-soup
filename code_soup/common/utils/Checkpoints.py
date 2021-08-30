import torch


class Checkpoints:
    @classmethod
    def save(self, PATH, model):
        torch.save(model.state_dict(), PATH)

    @classmethod
    def load(self, PATH, model):
        model.load_state_dict(torch.load(PATH))
        return model
