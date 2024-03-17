import torch
from torch import nn
from exercise_blanks import *
import data_loader as dl


class Attention(torch.nn.Module):
    """
    calculate embedding of the sequence, usning attention mechanism
    """

    def __init__(self, dim_emmbeding) -> None:
        super(Attention, self).__init__()
        self.v = nn.Linear(dim_emmbeding, dim_emmbeding)
        self.q = nn.Linear(dim_emmbeding, dim_emmbeding)
        self.r = nn.Linear(dim_emmbeding, dim_emmbeding)
        self.dim_emmbeding = dim_emmbeding
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        x: tensor of shape (batch, len_sequnce, dim_emmbeding)
        """
        # calculate attention weights
        v = self.v(x)
        q = self.q(x)
        r = self.r(x)
        attention = self.softmax(torch.bmm(q, v.transpose(1, 2)))
        # calculate embedding
        embedding = torch.bmm(attention, r)
        return embedding


class LogLinearAttention(torch.nn.Module):
    def __init__(self, dim_emmbeding) -> None:
        super(LogLinearAttention, self).__init__()
        self.attention = Attention(dim_emmbeding)
        self.linear = nn.Linear(dim_emmbeding, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: tensor of shape (batch, len_sequnce, dim_emmbeding)
        """
        embedding = self.attention(x).sum(dim=1)
        output = self.linear(embedding)
        return self.sigmoid(output)

    def predict(self, x):
        return (self.forward(x) > 0.5).int()


def train_attention(device="cpu"):

    # Set batch size
    batch_size = 64

    # Create DataManager object
    data_manager = DataManager(
        batch_size=batch_size,
        data_type=W2V_SEQUENCE,
        dataset_path="ex3\\stanfordSentimentTreebank",
        embedding_dim=300,
    )  # Initialize with your DataManager object

    # Initialize your log linear model with one-hot representation
    model = (
        LogLinearAttention(data_manager.get_input_shape()).to(torch.float64).to(device)
    )

    # Set hyperparameters
    n_epochs = 10
    lr = 0.01
    weight_decay = 0.01

    # Create data iterators with the specified batch size
    # Train the model
    train_record_data = train_model(
        model, data_manager, n_epochs, lr, weight_decay, device=device
    )
    test_loss, test_acc = evaluate(
        model,
        data_manager.get_torch_iterator(TEST),
        torch.nn.BCEWithLogitsLoss(),
        device=device,
    )
    print(f"test_loss{test_loss:.3f}, test_acc{test_acc:.3f}")

    return train_record_data


if __name__ == "__main__":
    rd = train_attention()
    rd.to_csv("ex3\\attention.csv")
