import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def weights_init_normal(m):
    """Initializes the weight and bias of the model.

    Args:
        m: A torch model to initialize.

    Returns:
        None.
    """

    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.2)
        torch.nn.init.constant_(m.bias.data, 0)


class DiscriminatorF(nn.Module):
    """FR-Train fairness discriminator.

    This class is for defining structure of FR-Train fairness discriminator.
    (ref: FR-Train paper, Section 3)

    Attributes:
        model: A model consisting of torch components.
    """

    def __init__(self, num_classes):
        """Initializes DiscriminatorF with torch components."""

        super(DiscriminatorF, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(num_classes, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_data):
        """Defines a forward operation of the model.

        Args:
            input_data: The input data.

        Returns:
            The predicted sensitive attribute for the given input data.
        """

        predicted_z = self.model(input_data)
        return predicted_z
