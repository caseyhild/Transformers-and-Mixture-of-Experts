import functools
from typing import Callable, Tuple, List

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("auto_diff.py"), '..')))

import auto_diff as ad
import torch
from torchvision import datasets, transforms

import math

max_len = 28

def linear_layer(input_node: ad.Node, weight: ad.Node, bias: ad.Node) -> ad.Node:
    """
    Compute the linear transformation: output = input @ weight + bias.

    Parameters
    ----------
    input_node : ad.Node
        Input node with shape (batch_size, input_features).
    weight : ad.Node
        Weight node with shape (input_features, output_features).
    bias : ad.Node
        Bias node with shape (output_features).

    Returns
    -------
    output : ad.Node
        The output node after applying the linear transformation.
    """
    
    # Compute output = input @ weight + bias
    return ad.add(ad.matmul(input_node, weight), bias)

def single_head_attention(input_node: ad.Node, W_Q: ad.Node, W_K: ad.Node, W_V: ad.Node, d_k: float) -> ad.Node:
    """
    Compute scaled dot-product attention for single-head attention mechanism.

    Parameters
    ----------
    input_node : ad.Node
        Input node with shape (batch_size, sequence_length, input_dim).
    W_Q : ad.Node
        Weight matrix for computing queries with shape (input_dim, d_k).
    W_K : ad.Node
        Weight matrix for computing keys with shape (input_dim, d_k).
    W_V : ad.Node
        Weight matrix for computing values with shape (input_dim, d_v).
    d_k : float
        Dimension of the keys.

    Returns
    -------
    output : ad.Node
        The output node after applying attention with shape (batch_size, sequence_length, d_v).
    """
    
    # Compute Q, K, and V from the input node and weights
    Q = ad.matmul(input_node, W_Q)
    K = ad.matmul(input_node, W_K)
    V = ad.matmul(input_node, W_V)
    
    # Compute the scaled dot-product attention weights A = Softmax(Q @ K^T / sqrt(d_k))
    A = ad.softmax(ad.div_by_const(ad.matmul(Q, ad.transpose(K, -2, -1)), math.sqrt(d_k)), dim=-1)
    
    # Return the matrix product of the attention weights, A and the value matrix, V
    return ad.matmul(A, V)

def encoder_layer(input_node: ad.Node, nodes: List[ad.Node],
                  model_dim: int, seq_length: int, eps: float) -> ad.Node:
    
    """
    Build an encoder layer that combines self-attention and feed-forward networks

    Parameters
    ----------
    input_node : ad.Node
        Input node with shape (batch_size, sequence_length, model_dim).
    nodes : List[ad.Node]
        A list to which all parameter nodes will be added.
    model_dim : int
        Hidden dimension (and the input dimension).
    seq_length : int
        Length of the input sequence.
    eps : float
        A small epsilon value for numerical stability in layernorm.

    Returns
    -------
    output : ad.Node
        A node with shape (batch_size, seq_length, model_dim) after processing
        through self-attention and feed-forward sub-layers with residual connections
        and layer normalization.
    """
    # Self-attention sub-layer
    
    # Create new nodes for Q, K, and V weights
    W_Q = ad.Variable('W_Q')
    W_K = ad.Variable('W_K')
    W_V = ad.Variable('W_V')
    nodes.extend([W_Q, W_K, W_V])
    
    # Create new node for O weights (output projection matrix)
    W_O = ad.Variable('W_O')
    nodes.append(W_O)
    
    # Compute self-attention (single-head attention)
    attention_to_input = single_head_attention(input_node, W_Q, W_K, W_V, model_dim)
    
    # Project the attention output
    attention_projection = ad.matmul(attention_to_input, W_O)
    
#     # Residual connection
#     attention_residual = ad.add(input_node, attention_projection)
    
    # Layer normalization
    attention_layer_output = ad.layernorm(attention_projection, normalized_shape=[model_dim], eps=eps)
    
    # Feed forward sub-layer (1 hidden layer)
    
    # Create new nodes for weights and biases of this layer (2 sets of weights and biases for 1 hidden layer)
    W_1 = ad.Variable('W_1')
    b_1 = ad.Variable('b_1')
    W_2 = ad.Variable('W_2')
    b_2 = ad.Variable('b_2')
    nodes.extend([W_1, b_1, W_2, b_2])
    
    # Define feed forward layer with 2 linear layers and a relu activation function
    hidden_layer = ad.relu(linear_layer(attention_layer_output, W_1, b_1))
    feed_forward_output = linear_layer(hidden_layer, W_2, b_2)
    
#     # Residual connection
#     feed_forward_residual = ad.add(attention_layer_output, feed_forward_output)
    
    # Layer normalization
    feed_forward_layer_output = ad.layernorm(feed_forward_output, normalized_shape=[model_dim], eps=eps)
    
    return feed_forward_layer_output

def transformer(X: ad.Node, nodes: List[ad.Node], 
                      model_dim: int, seq_length: int, eps, batch_size, num_classes) -> ad.Node:
    """Construct the computational graph for a single transformer layer with sequence classification.

    Parameters
    ----------
    X: ad.Node
        A node in shape (batch_size, seq_length, model_dim), denoting the input data.
    nodes: List[ad.Node]
        Nodes you would need to initialize the transformer.
    model_dim: int
        Dimension of the model (hidden size).
    seq_length: int
        Length of the input sequence.

    Returns
    -------
    output: ad.Node
        The output of the transformer layer, averaged over the sequence length for classification, in shape (batch_size, num_classes).
    """

    # Pass the input through an encoder layer
    encoder_layer_output = encoder_layer(X, nodes, model_dim, seq_length, eps)
    
    # Pool encoder layer output (average over sequence dimension)
    logits = ad.mean(encoder_layer_output, dim=(1,), keepdim=False)
    
    return logits

def softmax_loss(Z: ad.Node, y_one_hot: ad.Node, batch_size: int) -> ad.Node:
    """Construct the computational graph of average softmax loss over
    a batch of logits.

    Parameters
    ----------
    Z: ad.Node
        A node in of shape (batch_size, num_classes), containing the
        logits for the batch of instances.

    y_one_hot: ad.Node
        A node in of shape (batch_size, num_classes), containing the
        one-hot encoding of the ground truth label for the batch of instances.

    batch_size: int
        The size of the mini-batch.

    Returns
    -------
    loss: ad.Node
        Average softmax loss over the batch.
        When evaluating, it should be a zero-rank array (i.e., shape is `()`).

    Note
    ----
    1. In this homework, you do not have to implement a numerically
    stable version of softmax loss.
    2. You may find that in other machine learning frameworks, the
    softmax loss function usually does not take the batch size as input.
    Try to think about why our softmax loss may need the batch size.
    """

    # Compute log softmax of logits over class dimension
    log_softmax = ad.log(ad.softmax(Z, dim=1))
    
    # Sum the log_softmax over the classes (multiply by y_one_hot to only get log softmax of the corresponding classes)
    # i.e. Compute the negative log-likelihood of each sample
    loss_per_sample = ad.mul_by_const(ad.sum_op(ad.mul(y_one_hot, log_softmax), dim=(1,), keepdim=False), -1)
    
    # Compute and return the total loss over the entire batch
    loss = ad.div_by_const(ad.sum_op(loss_per_sample, dim=(0,), keepdim=False), batch_size)
    return loss

def sgd_epoch(
    f_run_model: Callable,
    X: torch.Tensor,
    y: torch.Tensor,
    model_weights: List[torch.Tensor],
    batch_size: int,
    lr: float,
) -> List[torch.Tensor]:
    """Run an epoch of SGD for the logistic regression model
    on training data with regard to the given mini-batch size
    and learning rate.

    Parameters
    ----------
    f_run_model: Callable
        The function to run the forward and backward computation
        at the same time for logistic regression model.
        It takes the training data, training label, model weight
        and bias as inputs, and returns the logits, loss value,
        weight gradient and bias gradient in order.
        Please check `f_run_model` in the `train_model` function below.

    X: torch.Tensor
        The training data in shape (num_examples, in_features).

    y: torch.Tensor
        The training labels in shape (num_examples,).

    model_weights: List[torch.Tensor]
        The model weights in the model.

    batch_size: int
        The mini-batch size.

    lr: float
        The learning rate.

    Returns
    -------
    model_weights: List[torch.Tensor]
        The model weights after update in this epoch.

    b_updated: torch.Tensor
        The model weight after update in this epoch.

    loss: torch.Tensor
        The average training loss of this epoch.
    """

    num_examples = X.shape[0]
    num_batches = (num_examples + batch_size - 1) // batch_size  # Compute the number of batches
    total_loss = 0.0

    for i in range(num_batches):
        # Get the mini-batch data
        start_idx = i * batch_size
        if start_idx + batch_size > num_examples: continue
        end_idx = min(start_idx + batch_size, num_examples)
        X_batch = X[start_idx:end_idx, :max_len]
        y_batch = y[start_idx:end_idx]
        
        # Compute forward and backward passes

        # Run the model with current model_weights on the given batch
        logits, loss, *gradients = f_run_model(model_weights, X_batch, y_batch)
        
        # Update weights and biases
        # W_Q -= lr * grad_W_Q.sum(dim=0)
        
        # Update each model_weight by subtracting its gradient scaled by the learning rate
        # Sum gradients over batch
        for idx, weight in enumerate(model_weights):
            dims = tuple(range(3 - len(model_weights[idx].shape)))
            model_weights[idx] -= lr * gradients[idx].sum(dim=dims)

        # Add loss for this iteration to total loss
        total_loss += loss.item() * (end_idx - start_idx)

    # Compute the average loss
    
    average_loss = total_loss / num_examples
    print('Avg_loss:', average_loss)

    return model_weights, average_loss

def train_model():
    """Train a logistic regression model with handwritten digit dataset.

    Note
    ----
    Your implementation should NOT make changes to this function.
    """
    # Set up model params

    # Hyperparameters
    input_dim = 28  # Each row of the MNIST image
    seq_length = max_len  # Number of rows in the MNIST image
    num_classes = 10 #
    model_dim = 128 #
    eps = 1e-5

    # - Set up the training settings.
    num_epochs = 20
    batch_size = 50
    lr = 0.02

    # Define the forward graph.
    
    # Create input node X
    X = ad.Variable(name='X')
    
    # Run transformer model on input X
    nodes = []
    transformer_output = transformer(X, nodes, model_dim, seq_length, eps, batch_size, num_classes)
    
    # Get predicted and true outputs as well as loss
    y_predict: ad.Node = transformer_output
    y_groundtruth = ad.Variable(name="y")
    loss: ad.Node = softmax_loss(y_predict, y_groundtruth, batch_size)
    
    # Construct backward graph using gradients of forward graph
    gradients = ad.gradients(loss, nodes)
    
    # Create the evaluator
    grads: List[ad.Node] = gradients
    evaluator = ad.Evaluator([y_predict, loss, *grads])
    test_evaluator = ad.Evaluator([y_predict])

    # - Load the dataset.
    #   Take 80% of data for training, and 20% for testing.
    # Prepare the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    # Convert the train dataset to NumPy arrays
    X_train = train_dataset.data.numpy().reshape(-1, 28, 28) / 255.0  # Flatten to 784 features
    y_train = train_dataset.targets.numpy()

    # Convert the test dataset to NumPy arrays
    X_test = test_dataset.data.numpy().reshape(-1, 28, 28) / 255.0  # Flatten to 784 features
    y_test = test_dataset.targets.numpy()

    # Initialize the OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)  # Use sparse=False to get a dense array

    # Fit and transform y_train, and transform y_test
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))

    num_classes = 10

    # Initialize model weights.
    np.random.seed(0)
    stdv = 1.0 / np.sqrt(num_classes)
    W_Q_val = torch.from_numpy(np.random.uniform(-stdv, stdv, (input_dim, model_dim)))
    W_K_val = torch.from_numpy(np.random.uniform(-stdv, stdv, (input_dim, model_dim)))
    W_V_val = torch.from_numpy(np.random.uniform(-stdv, stdv, (input_dim, model_dim)))
    W_O_val = torch.from_numpy(np.random.uniform(-stdv, stdv, (model_dim, model_dim)))
    W_1_val = torch.from_numpy(np.random.uniform(-stdv, stdv, (model_dim, model_dim)))
    b_1_val = torch.from_numpy(np.random.uniform(-stdv, stdv, (model_dim,)))
    W_2_val = torch.from_numpy(np.random.uniform(-stdv, stdv, (model_dim, num_classes)))
    b_2_val = torch.from_numpy(np.random.uniform(-stdv, stdv, (num_classes,)))

    def f_run_model(model_weights, X_batch, y_batch):
        """The function to compute the forward and backward graph.
        It returns the logits, loss, and gradients for model weights.
        """
        result = evaluator.run(
            input_values={
                X: X_batch,
                y_groundtruth: y_batch,
                nodes[0]: W_Q_val,
                nodes[1]: W_K_val,
                nodes[2]: W_V_val,
                nodes[3]: W_O_val,
                nodes[4]: W_1_val,
                nodes[5]: b_1_val,
                nodes[6]: W_2_val,
                nodes[7]: b_2_val

            }
        )
        return result

    def f_eval_model(X_val, model_weights: List[torch.Tensor]):
        """The function to compute the forward graph only and returns the prediction."""
        num_examples = X_val.shape[0]
        num_batches = (num_examples + batch_size - 1) // batch_size  # Compute the number of batches
        total_loss = 0.0
        all_logits = []
        for i in range(num_batches):
            # Get the mini-batch data
            start_idx = i * batch_size
            if start_idx + batch_size > num_examples: continue
            end_idx = min(start_idx + batch_size, num_examples)
            X_batch = X_val[start_idx:end_idx, :max_len]
            logits = test_evaluator.run({
                X: X_batch,
                nodes[0]: W_Q_val,
                nodes[1]: W_K_val,
                nodes[2]: W_V_val,
                nodes[3]: W_O_val,
                nodes[4]: W_1_val,
                nodes[5]: b_1_val,
                nodes[6]: W_2_val,
                nodes[7]: b_2_val

            })
            all_logits.append(logits[0])
        # Concatenate all logits and return the predicted classes
        concatenated_logits = np.concatenate(all_logits, axis=0)
        predictions = np.argmax(concatenated_logits, axis=1)
        return predictions

    # Train the model.
    X_train, X_test, y_train, y_test= torch.tensor(X_train), torch.tensor(X_test), torch.DoubleTensor(y_train), torch.DoubleTensor(y_test)
    model_weights: List[torch.Tensor] = [W_Q_val, W_K_val, W_V_val, W_O_val,
                                         W_1_val, b_1_val, W_2_val, b_2_val]
    for epoch in range(num_epochs):
        X_train, y_train = shuffle(X_train, y_train)
        model_weights, loss_val = sgd_epoch(
            f_run_model, X_train, y_train, model_weights, batch_size, lr
        )

        # Evaluate the model on the test data.
        predict_label = f_eval_model(X_test, model_weights)
        print(
            f"Epoch {epoch}: test accuracy = {np.mean(predict_label== y_test.numpy())}, "
            f"loss = {loss_val}"
        )

    # Return the final test accuracy.
    predict_label = f_eval_model(X_test, model_weights)
    return np.mean(predict_label == y_test.numpy())


if __name__ == "__main__":
    print(f"Final test accuracy: {train_model()}")
