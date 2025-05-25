from typing import List
import numpy as np
import tensorflow as tf
from classes.Layer import Layer

class Embedding:
    def __init__(self, W: np.ndarray):
        self.W = W  # (vocab_size, emb_dim)

    def forward(self, X: np.ndarray) -> np.ndarray:
        # X: (batch, seq_len)
        return self.W[X]  # → (batch, seq_len, emb_dim)

class LSTM:
    def __init__(self,
                 kernel: np.ndarray,
                 recurrent: np.ndarray,
                 bias: np.ndarray,
                 units: int,
                 return_sequences: bool):
        self.W = kernel
        self.U = recurrent
        self.b = bias
        self.units = units
        self.return_sequences = return_sequences

    def forward(self, X: np.ndarray) -> np.ndarray:
        # input 2D → treat as seq_len=1
        if X.ndim == 2:
            X = X[:, None, :]  # → (batch,1,dim)
        batch, seq_len, _ = X.shape
        h = np.zeros((batch, self.units), dtype=X.dtype)
        c = np.zeros((batch, self.units), dtype=X.dtype)
        outputs = []
        for t in range(seq_len):
            x_t = X[:, t, :]  # (batch, in_dim)
            z   = x_t.dot(self.W) + h.dot(self.U) + self.b
            i   = 1/(1+np.exp(-z[:, :self.units]))
            f   = 1/(1+np.exp(-z[:, self.units:2*self.units]))
            c_bar = np.tanh(z[:, 2*self.units:3*self.units])
            o   = 1/(1+np.exp(-z[:, 3*self.units:]))
            c   = f*c + i*c_bar
            h   = o*np.tanh(c)
            outputs.append(h)
        if self.return_sequences:
            return np.stack(outputs, axis=1)  # (batch, seq_len, units)
        else:
            return h  # (batch, units)

class Bidirectional:
    def __init__(self,
                 fw: LSTM,
                 bw: LSTM,
                 return_sequences: bool):
        self.fw = fw
        self.bw = bw
        self.return_sequences = return_sequences

    def forward(self, X: np.ndarray) -> np.ndarray:
        # forward pass
        out_f = self.fw.forward(X)
        # backward on reversed time axis
        out_b = self.bw.forward(X[:, ::-1, :])
        if self.return_sequences:
            # both out_f/out_b must be 3D
            out_b = out_b[:, ::-1, :] 
            if out_f.ndim != 3 or out_b.ndim != 3:
                raise ValueError('Cannot return sequences when sublayers do not output sequences')
            # concat feature dim
            return np.concatenate([out_f, out_b], axis=2)  # (batch, seq_len, units*2)
        else:
            # take last timestep if 3D, else take as is
            if out_f.ndim == 3:
                out_f = out_f[:, -1, :]  # (batch, units)
            if out_b.ndim == 3:
                out_b = out_b[:, -1, :]
            return np.concatenate([out_f, out_b], axis=1)  # (batch, units*2)

def unpack(l):
    w = l.get_weights()
    if len(w)==3: return w
    k,u = w; b = np.zeros(4*l.units, dtype=k.dtype); return (k,u,b)

def build_pipeline(keras_model):
    layers = []
    # 1) Embedding
    emb_W = keras_model.get_layer('emb').get_weights()[0]
    layers.append(Embedding(emb_W))

    # 2) (Bi)LSTM
    for layer in keras_model.layers:
        from tensorflow.keras.layers import LSTM as KerasLSTM, Bidirectional as KerasBi
        if isinstance(layer, KerasBi):
            fw_k,fw_u,fw_b = unpack(layer.forward_layer)
            bw_k,bw_u,bw_b = unpack(layer.backward_layer)
            rs = layer.return_sequences
            layers.append(Bidirectional(
                LSTM(fw_k, fw_u, fw_b, layer.forward_layer.units, return_sequences=rs),
                LSTM(bw_k, bw_u, bw_b, layer.backward_layer.units, return_sequences=rs),
                return_sequences=rs
            ))
        elif isinstance(layer, KerasLSTM):
            k,u,b = unpack(layer)
            layers.append(LSTM(
                kernel=k,
                recurrent=u,
                bias=b,
                units=layer.units,
                return_sequences=layer.return_sequences
            ))
            
    # 3) Dense output (softmax)
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
    return a  # (batch, n_classes)

if __name__ == '__main__':
    print('Running unit tests for forward_lstm.py')

    # Embedding test
    W = np.arange(12).reshape(6,2)
    emb = Embedding(W)
    X = np.array([[1,5],[0,3]])
    out = emb.forward(X)
    assert out.shape == (2,2,2)
    assert np.all(out[1,1]==W[3])

    # LSTM test
    batch, seq, dim, units = 2,4,3,5
    X0 = np.zeros((batch,seq,dim))
    lstm_seq = LSTM(np.zeros((dim,4*units)),
                    np.zeros((units,4*units)),
                    np.zeros((4*units,)),
                    units, return_sequences=True)
    o_seq = lstm_seq.forward(X0)
    assert o_seq.shape==(batch,seq,units)

    lstm_last = LSTM(np.zeros((dim,4*units)),
                     np.zeros((units,4*units)),
                     np.zeros((4*units,)),
                     units, return_sequences=False)
    o_last = lstm_last.forward(X0)
    assert o_last.shape==(batch,units)

    # Bidirectional test
    bi_last = Bidirectional(lstm_seq, lstm_seq, return_sequences=False)
    b_o = bi_last.forward(X0)
    assert b_o.shape==(batch,units*2)

    bi_seq = Bidirectional(lstm_seq, lstm_seq, return_sequences=True)
    b_o2 = bi_seq.forward(X0)
    assert b_o2.shape==(batch,seq,units*2)

    # Dense test
    inp = np.zeros((batch,units*2))
    dense = Layer(units*2,3,activation='softmax',weight_init='zero')
    dense.W = np.zeros((units*2,3)); dense.b = np.zeros((1,3))
    d_out = dense.forward(inp)
    assert d_out.shape==(batch,3)
    assert np.allclose(d_out.sum(axis=1),1)

    print('All unit tests passed!')
