from torch import nn

class VAE(nn.Module):

  hidden_dims = [32, 64]

  def __init__(self, channels_in: int):
    super(VAE, self).__init__()

    modules = []
    channels = channels_in
    for h_dim in self.hidden_dims:
      modules.append(
        nn.Conv2d(
          channels, # Number of channels in the previous layer.
          h_dim, # Number of channels in this layer.
          3, # Kernel size.
          stride='same',
          padding=1,
        )
      )
      channels = h_dim

    self.encoder = nn.Sequential(*modules)
