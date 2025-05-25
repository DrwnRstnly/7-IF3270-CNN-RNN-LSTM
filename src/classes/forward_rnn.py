from typing import List
import numpy as np
import tensorflow as tf
from classes.Layer import Layer, tanh
from tensorflow.keras.models import Model
from tensorflow.keras.layers import SimpleRNN as KerasSRNN
from tensorflow.keras.layers import Bidirectional as KerasBi

class Embedding:
    def __init__(self, W: np.ndarray):
        self.W = W  # (vocab_size, emb_dim)

    def forward(self, X: np.ndarray) -> np.ndarray:
        # X: (batch, seq_len)
        return self.W[X]  # → (batch, seq_len, emb_dim)

class RNN:
    def __init__(self,
                 kernel: np.ndarray,
                 recurrent: np.ndarray,
                 b_xh: np.ndarray,
                 units: int,
                 return_sequences: bool):
        self.W = kernel # input to hidden (input_dims, units)
        self.U = recurrent # hidden to hidden (units, units)
        self.b_xh = b_xh # bias hidden
        self.units = units # number of neuron hidden layer
        self.return_sequences = return_sequences

    def forward(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 2: #always make sure it is in 3d
            X = X[:, None, :] # (batch, 1, dim)
        
        # batch: number of independent paralel process
        # seq_len: length of example sequence
        batch, seq_len, _ = X.shape

        h = np.zeros((batch, self.units), dtype=X.dtype)

        outputs = []

        # loop each timestep
        for t in range (seq_len):
            # data for each batch for the specific timestep
            x_t = X[:, t, :] # (batch, in_dim)
            h = tanh(x_t.dot(self.W) + h.dot(self.U) + self.b_xh) # defaultnya keras, tanh
            outputs.append(h)

        if (self.return_sequences):
            return np.stack(outputs, axis = 1)
        else:
            return outputs[-1]

class RNNBidirectional:
    def __init__(self,
            fw: RNN,
            bw: RNN,
            return_sequences: bool):
        self.fw = fw
        self.bw = bw
        self.return_sequences = return_sequences

    def forward(self, X: np.ndarray) -> np.ndarray:
        output_forward = self.fw.forward(X)
        output_backward = self.bw.forward(X[:, ::-1, :])

        if self.return_sequences:
            if output_forward.ndim != 3 or output_backward.ndim != 3:
                raise ValueError(
                    "Cannot return sequences when sublayers do not output sequences"
                )
            
            output_backward = output_backward[:, ::-1, :]
            return np.concatenate([output_forward, output_backward], axis = 2)
        else:
            if output_forward.ndim == 3:
                output_forward = output_forward[:, -1, :]
            if output_backward.ndim == 3:
                output_backward = output_backward[:, -1, :]
            return np.concatenate([output_forward, output_backward], axis = 1)
        
def build_pipeline(keras_model: Model):
    layers = []
    
    # Embedding layer
    emb_W = keras_model.get_layer('emb').get_weights()[0]
    layers.append(Embedding(emb_W))

    # Bi/Uni-directional RNN
    for layer in keras_model.layers:
        if (isinstance(layer, KerasBi)):
            rnn_layer = layer.forward_layer
            fw_kernel, fw_recurrent, fw_bias = rnn_layer.get_weights()
            bw_kernel, bw_recurrent, bw_bias = layer.backward_layer.get_weights()

            fw = RNN(
                kernel=fw_kernel,
                recurrent=fw_recurrent,
                b_xh=fw_bias[:rnn_layer.units],
                units=rnn_layer.units,
                return_sequences=layer.return_sequences,
            )

            bw = RNN(
                kernel=bw_kernel,
                recurrent=bw_recurrent,
                b_xh=bw_bias[:rnn_layer.units],
                units=rnn_layer.units,
                return_sequences=layer.return_sequences,
            )
            layers.append(RNNBidirectional(fw, bw, return_sequences=layer.return_sequences))

        elif (isinstance(layer, KerasSRNN)):
            W, U, b = layer.get_weights()
            units = layer.units
            layers.append(RNN(
                kernel=W,
                recurrent=U,
                b_xh=b[:units],
                units=units,
                return_sequences=layer.return_sequences
            ))  

    # Dense (softmax)
    for layer in keras_model.layers[::-1]:
        if isinstance(layer, tf.keras.layers.Dense) and layer.activation.__name__=='softmax':
            Wd, bd = layer.get_weights()
            d = Layer(Wd.shape[0], Wd.shape[1], activation='softmax', weight_init='zero')
            d.W = Wd; d.b = bd.reshape(1, -1)
            layers.append(d)
            break
    return layers

def predict(pipeline: List[Layer], X_tok: np.ndarray) -> np.ndarray:
    a = pipeline[0].forward(X_tok)
    for layer in pipeline[1:]:
        a = layer.forward(a)
    return a

        



        
        