import torch as th
import torch.nn as nn


class MotorRNN(nn.Module):
    """
    RNN module class utilising a network with 1 layer of Gated Recurrent Units, to be extended by the Monolothic & Dual Networks (Task and Predictive Networks). The GRU is chosen for its gating mechanisms that help mitigate vanishing gradient issues, making it suitable for learning long-term dependencies in sequential data, which is crucial for our delayed feedback tasks. The class includes orthogonal weight initialisation to further stabilise training and memory retention across the feedback delay period.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        device: th.device,
        is_task_network: bool,
        effector_type: str
    ):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.effector_type = effector_type

        # GRU Core: maintains an internal hidden state vector and at each time step, it integrates it with the new sensory observation. Biologically, this allows contextualising current sensory feedback against past motor commands.
        self.gru = nn.GRU(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True
        )

        # (Fully-Connected) Linear Readout: projects the GRU high-dimensional latent space down to the specific dimensionality required by the physical task.
        self.fc = nn.Linear(in_features=hidden_dim, out_features=output_dim)

        self.is_task_network = is_task_network
        if self.is_task_network:
            self.activation = nn.Sigmoid()  # Bounds muscle activations between 0 and 1, needed only for the Task Network which directly controls the muscles. The Predictive Network outputs can be unbounded as they are not directly controlling the muscles.

        self._initialise_weights()
        self.to(device) 

    def _initialise_weights(self):
        """Applies orthogonal initialisation (forces eigenvalues of the recurrent matrix to be 1) to stabilise memory retention and BPTT and avoid exploding or vanishing gradients over the feedback delay period. 
        
        Xavier initialisation is applied to the input-to-hidden and fully-connected matrices to ensure the variance of the forward signal remains consistent as it passes through the network layers, preventing the signal from saturating. 
        
        Biases are initialised to zero, except for the Task Network's output layer bias which is initialised to a negative value to prevent violent initial muscle spasms."""
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                # Hidden-to-hidden matrix: dictates how heavily the previous internal state of the network updates the current internal state, thus governing memory retention across time.
                nn.init.orthogonal_(param)
            elif "weight_ih" in name or "fc.weight" in name:
                # Input-to-hidden matrix: dictates how heavily the current incoming sensory data vector updates the network's internal state.
                # ih and fc matrices do not recurr over time, they map data from one dimension to another: Xavier initialisation draws random numbers from a uniform distribution scaled by the number of input and output neurons in that specific layer. This ensures the variance of the forward signal remains consistent as it passes through the network layers, preventing the signal from saturating.
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                if self.is_task_network and "fc.bias" in name:
                    # If rigid_arm26, rely on PyTorch's default uniform initialisation: baseline muscle activation of 50%. For a compliant tendon, this would trigger the tendons to violently snap, causing gradient explosions.
                    if self.effector_type == 'compliant_arm26':
                        # Initialise task bias negatively to prevent violent initial muscle spasms. Adjusted from -5.0 (less than 1% tone) to -2.2 to provide 10% baseline tone and prevent compliant tendon slack and vanishing gradients to avoid Initialisation Paralysis
                        nn.init.constant_(param, -2.2) 
                    else: 
                        nn.init.constant_(param, -5.0)   # ~0.7% tone: matches the validated rigid baseline
                else:
                    nn.init.zeros_(param)

    def forward(self, x: th.Tensor, h_prev: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """Defines the forward computational graph. The GRU processes the input sequence and previous hidden state to produce an output and a new hidden state. The output is then passed through a fully connected layer to project it down to the required output dimensionality. If this is the Task Network, a sigmoid activation function is applied to bound the muscle activations between 0 and 1."""
        
        # MotorNet’s env.step() outputs an observation for a single moment in time so x shape is (batch_size, features) -> transformed to (batch_size, sequence_length=1, features)
        y, h_new = self.gru(x.unsqueeze(1), h_prev)

        out = self.fc(y).squeeze(1)
        if self.is_task_network:
            out = self.activation(out)
        return out, h_new

    def init_hidden(self, batch_size: int) -> th.Tensor:
        """Generates a zeroed hidden state tensor for a new reaching trial. The shape is (num_layers, batch_size, hidden_dim) where num_layers=1 in our case."""
        return th.zeros(1, batch_size, self.hidden_dim, device=self.device)
