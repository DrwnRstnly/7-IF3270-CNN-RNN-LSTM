from classes.Layer import Layer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle 

# Loss Functions & Derivatives

def mse_loss(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

def mse_loss_deriv(y_true, y_pred):
    return (y_pred - y_true) / y_true.shape[0]

def binary_crossentropy_loss(y_true, y_pred, epsilon=1e-12):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

def binary_crossentropy_loss_deriv(y_true, y_pred, epsilon=1e-12):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    grad = (-y_true / y_pred + (1 - y_true) / (1 - y_pred)) / y_true.shape[0]
    return grad

def categorical_crossentropy_loss(y_true, y_pred, epsilon=1e-12):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    return loss

def categorical_crossentropy_loss_deriv(y_true, y_pred):
    return (y_pred - y_true) / y_true.shape[0]

#  Neural Network Class

class NeuralNetwork:
    def __init__(self, input_size, layers_config, loss_function='mse', 
                 weight_init='random_uniform', weight_init_params={}, regularization=None, reg_lambda=0.0,
                 use_rmsnorm=False, rmsnorm_epsilon=1e-8):
        self.layers = []
        prev_size = input_size
        for neurons, activation in layers_config:
            layer = Layer(prev_size, neurons, activation, weight_init, weight_init_params, use_rmsnorm=use_rmsnorm, rmsnorm_epsilon=rmsnorm_epsilon)
            self.layers.append(layer)
            prev_size = neurons
        
        self.loss_name = loss_function.lower()
        if self.loss_name == 'mse':
            self.loss = mse_loss
            self.loss_deriv = mse_loss_deriv
        elif self.loss_name == 'binary_crossentropy':
            self.loss = binary_crossentropy_loss
            self.loss_deriv = binary_crossentropy_loss_deriv
        elif self.loss_name == 'categorical_crossentropy':
            self.loss = categorical_crossentropy_loss
            self.loss_deriv = categorical_crossentropy_loss_deriv
        else:
            raise ValueError("Unknown loss function.")
        
        self.regularization = regularization # 'L1' atau 'L2' atau None
        self.reg_lambda = reg_lambda

        self.use_rmsnorm = use_rmsnorm
        self.epsilon = rmsnorm_epsilon

    def forward(self, X):
        a = X
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def backward(self, X, y, learning_rate):
        output = self.forward(X)
        delta = self.loss_deriv(y, output)
        for layer in reversed(self.layers):
            delta = layer.backward(delta, learning_rate, reg_type=self.regularization, reg_lambda=self.reg_lambda)

    def train(self, X_train, y_train, X_val, y_val, batch_size=32, learning_rate=0.01, max_epoch=100, verbose=1):
        n_samples = X_train.shape[0]
        history = {"train_loss": [], "val_loss": []}
        for epoch in range(max_epoch):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                self.backward(X_batch, y_batch, learning_rate)

            y_pred = self.forward(X_train)
            y_pred_val = self.forward(X_val)
            train_loss = self.loss(y_train, y_pred)
            val_loss = self.loss(y_val, y_pred_val)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            if verbose:
                print(f"Epoch {epoch+1}/{max_epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}")
        return history
    
    def predict(self, X):
        return self.forward(X)
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filename}.")

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filename}.")
        return model
    
    def display_model(self):
        print("Model Structure:")
        for idx, layer in enumerate(self.layers):
            print(f"\nLayer {idx+1}:")
            print(f"  Input size: {layer.input_size}, Output size: {layer.output_size}")
            print(f"  Activation: {layer.activation_name}")
            print("  Weights:\n", layer.W)
            print("  Biases:\n", layer.b)
            if layer.dW is not None:
                print("  Weight Gradients (dW):\n", layer.dW)
                print("  Bias Gradients (db):\n", layer.db)
            else:
                print("  Gradients not computed yet.")

    def visualize_network(self, figsize=(20, 12), max_neurons_to_show=None, weight_alpha=0.05, neuron_size=100, 
                        gradient_cmap='coolwarm', weight_cmap='viridis', edge_threshold=None, 
                        sample_weights=True, sample_ratio=0.1):
        
        # Create a default max_neurons if not provided
        if max_neurons_to_show is None:
            max_neurons_to_show = {}
            for i, layer in enumerate([self.layers[0].input_size] + [layer.output_size for layer in self.layers]):
                if layer > 100:
                    max_neurons_to_show[i] = 20
                elif layer > 50:
                    max_neurons_to_show[i] = 15
                else:
                    max_neurons_to_show[i] = layer
        
        layer_sizes = [self.layers[0].input_size] + [layer.output_size for layer in self.layers]
        n_layers = len(layer_sizes)
        
        fig, ax = plt.subplots(figsize=figsize)
    
        x_spacing = 1.0 / (n_layers - 1)
        
        neuron_positions = {}
        
        # Draw neurons for each layer
        for layer_idx, size in enumerate(layer_sizes):
            neurons_to_show = min(size, max_neurons_to_show.get(layer_idx, size))
            
            if size > neurons_to_show:
                if layer_idx == 0:
                    print(f"Input layer: showing {neurons_to_show} of {size} neurons")
                else:
                    print(f"Layer {layer_idx}: showing {neurons_to_show} of {size} neurons")
                indices = np.sort(np.random.choice(size, neurons_to_show, replace=False))
            else:
                indices = np.arange(size)
            
            y_spacing = 1.0 / (neurons_to_show + 1)
            for i, idx in enumerate(indices):
                y = (i + 1) * y_spacing
                circle = plt.Circle((layer_idx * x_spacing, y), 0.02, 
                                fill=True, color='lightblue', edgecolor='blue', zorder=4)
                ax.add_patch(circle)
                
                if size <= 50 or i % 5 == 0:  
                    ax.text(layer_idx * x_spacing, y, f"{idx}", ha='center', va='center', 
                        fontsize=8, zorder=5)
                
                neuron_positions[(layer_idx, idx)] = (layer_idx * x_spacing, y)
        
        # Draw connections (weights) between layers
        for layer_idx, layer in enumerate(self.layers):
            source_layer_idx = layer_idx
            target_layer_idx = layer_idx + 1
            
            weights = layer.W
            
            if hasattr(layer, 'dW') and layer.dW is not None:
                weight_gradients = layer.dW
            else:
                weight_gradients = np.zeros_like(weights)
            
            # Normalize weights and gradients for color mapping
            max_weight = np.max(np.abs(weights)) if weights.size > 0 else 1
            max_gradient = np.max(np.abs(weight_gradients)) if weight_gradients.size > 0 else 1
            
            weight_norm = plt.Normalize(-max_weight, max_weight)
            gradient_norm = plt.Normalize(-max_gradient, max_gradient)
            
            source_indices = list(neuron_positions.keys())
            source_indices = [idx for idx in source_indices if idx[0] == source_layer_idx]
            
            target_indices = list(neuron_positions.keys())
            target_indices = [idx for idx in target_indices if idx[0] == target_layer_idx]
            
            # Calculate how many connections to sample
            total_connections = len(source_indices) * len(target_indices)
            if sample_weights and total_connections > 1000:
                sample_size = int(total_connections * sample_ratio)
                connections_to_show = min(total_connections, sample_size)
                print(f"Layer {layer_idx+1}: showing {connections_to_show} of {total_connections} connections")
                
                source_targets = []
                for _ in range(connections_to_show):
                    source_idx = source_indices[np.random.randint(0, len(source_indices))]
                    target_idx = target_indices[np.random.randint(0, len(target_indices))]
                    source_targets.append((source_idx, target_idx))
            else:
                source_targets = [(s, t) for s in source_indices for t in target_indices]
            
            # Draw connections
            for (source_layer, source_neuron), (target_layer, target_neuron) in source_targets:
                weight = weights[source_neuron, target_neuron]
                
                if edge_threshold is not None and abs(weight) < edge_threshold:
                    continue
                    
                gradient = weight_gradients[source_neuron, target_neuron]
                
                start = neuron_positions[(source_layer, source_neuron)]
                end = neuron_positions[(target_layer, target_neuron)]
                
                line_width = min(5, 0.5 + abs(weight) / max_weight * 3)
                
                weight_color = cm.get_cmap(weight_cmap)(weight_norm(weight))
                gradient_color = cm.get_cmap(gradient_cmap)(gradient_norm(gradient))
                
                ax.plot([start[0], end[0]], [start[1], end[1]], 
                    color=weight_color, linewidth=line_width, alpha=weight_alpha, zorder=1)
                
                if np.abs(gradient) > max_gradient * 0.1: 
                    mid_x = (start[0] + end[0]) / 2
                    mid_y = (start[1] + end[1]) / 2
                    gradient_size = min(20, 5 + (abs(gradient) / max_gradient) * 15)
                    ax.scatter(mid_x, mid_y, s=gradient_size, color=gradient_color, 
                            edgecolor='black', linewidth=0.5, alpha=0.7, zorder=3)
        
    
        legend_x = 0
        legend_y = 1
        legend_spacing = 0.06
        
        
        # Weight legend 
        weight_cax = fig.add_axes([legend_x, legend_y + legend_spacing, 0.2, 0.02])
        weight_cb = fig.colorbar(cm.ScalarMappable(norm=weight_norm, cmap=weight_cmap), 
                            cax=weight_cax, orientation='horizontal')
        weight_cb.set_label('Weights', fontsize=8)
        weight_cb.ax.tick_params(labelsize=6)
        
        # Gradient legend
        grad_cax = fig.add_axes([legend_x, legend_y + 2*legend_spacing, 0.2, 0.02])
        grad_cb = fig.colorbar(cm.ScalarMappable(norm=gradient_norm, cmap=gradient_cmap), 
                            cax=grad_cax, orientation='horizontal')
        grad_cb.set_label('Gradients', fontsize=8)
        grad_cb.ax.tick_params(labelsize=6)
        
        for i, size in enumerate(layer_sizes):
            if i == 0:
                layer_name = f"Input Layer\n{size} neurons"
            elif i == len(layer_sizes) - 1:
                layer_name = f"Output Layer\n{size} neurons"
            else:
                layer_name = f"Hidden Layer {i}\n{size} neurons"
            ax.text(i * x_spacing, 1.05, layer_name, ha='center', va='center', fontsize=12)

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.1)
        ax.axis('off')
        
        plt.tight_layout()
        return fig, ax
    
    def plot_weight_distribution(self, layers_to_plot):
        for idx in layers_to_plot:
            if idx < 0 or idx >= len(self.layers):
                print(f"Layer index {idx} is out of range.")
                continue
            plt.figure()
            plt.hist(self.layers[idx].W.flatten(), bins=20, color="blue", label='Weights')
            plt.hist(self.layers[idx].b.flatten(), bins=20, color="yellow", label='Biases')
            plt.legend()
            plt.title(f"Weight and Bias Distribution for Layer {idx+1}")
            plt.xlabel("Weight value")
            plt.ylabel("Frequency")
            plt.show()

    def plot_gradient_distribution(self, layers_to_plot):
        for idx in layers_to_plot:
            if idx < 0 or idx >= len(self.layers):
                print(f"Layer index {idx} is out of range.")
                continue
            if self.layers[idx].dW is None:
                print(f"No gradient available for Layer {idx+1}. Run a backward pass first.")
                continue
            plt.figure()
            plt.hist(self.layers[idx].dW.flatten(), bins=20,color="blue", label='dW')
            plt.hist(self.layers[idx].db.flatten(), bins=20,color="yellow", label='db')
            plt.legend()
            plt.title(f"Gradient Distribution (dW) for Layer {idx+1}")
            plt.xlabel("Gradient value")
            plt.ylabel("Frequency")
            plt.show()

    def plot_training_loss(self, history):
        epochs = range(1, len(history["train_loss"]) + 1)

        plt.figure(figsize=(10, 6))

        plt.plot(epochs, history["train_loss"], label="Training Loss", marker="o")
        
        plt.plot(epochs, history["val_loss"], label="Validation Loss", marker="x")

        plt.title("Grafik Training dan Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        
        plt.legend()
        plt.grid(True)

        plt.show()
