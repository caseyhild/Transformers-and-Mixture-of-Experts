from typing import Any, Dict, List
import torch
from auto_diff import *

class MatMulLayerNormOp(Op):
    """Fused matrix multiplication and layer normalization operation."""

    def __call__(
        self, 
        node_A: Node, 
        node_B: Node, 
        normalized_shape: List[int], 
        eps: float = 1e-5
    ) -> Node:
        """
        Args:
            node_A: The first input node.
            node_B: The second input node.
            normalized_shape: The shape of the normalization axes.
            eps: The epsilon value to avoid division by zero.
        """
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={
                "normalized_shape": normalized_shape,
                "eps": eps
            },
            name=f"MatMulLayerNorm({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and layer normalization result."""
        assert len(input_values) == 2
        
        # Compute matmul
        
        matmul_result = None
        if input_values[0].ndim == 2:
            # Compute 2D matrix multiplication
            A = input_values[0]
            B = input_values[1]
            assert A.shape[1] == B.shape[0]
            matmul_result = A @ B
        else:
            # Compute 3D matrix multiplication, a series of 2D matrix multiplications (often a 2D matmul for each element in batch)
            results = []
            for i in range(len(input_values[0])):
                A = input_values[0][i]
                if input_values[1].ndim == 2:
                    B = input_values[1]
                else:
                    B = input_values[1][i]
                assert A.shape[1] == B.shape[0]
                results.append(A @ B)
            matmul_result = torch.stack(results, dim=0)
        
        # Compute layernorm
        
        normalized_dims = list(range(matmul_result.ndim - len(node.normalized_shape), matmul_result.ndim))
        
        # Calculate the mean and standard deviation
        mu = matmul_result.mean(dim=normalized_dims, keepdim=True)
        std = torch.sqrt(((matmul_result - mu) ** 2).mean(dim=normalized_dims, keepdim=True) + node.eps)
        
        # Return the normalized version of the input: (input - mean) / std
        return (matmul_result - mu) / std

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        
        # Compute gradient of layernorm
        
        # Get matmul result
        matmul_result = matmul(node.inputs[0], node.inputs[1])
        
        # Get normalized dimensions
        normalized_dims = list(range(-len(node.normalized_shape), 0))
        
        # Calculate mean and standard deviation
        mu = mean(matmul_result, dim=normalized_dims, keepdim=True)
        std = sqrt(mean((matmul_result - mu) * (matmul_result - mu), dim=normalized_dims, keepdim=True) + node.eps)
        
        # Calculate gradient of mean and gradient of difference of input from mean
        d_mu = mean(output_grad, dim=normalized_dims, keepdim=True)
        d_input_mu = mean((output_grad * (matmul_result - mu)), dim=normalized_dims, keepdim=True)
        
        # Return overall layer norm gradient with respect to input
        layernorm_grad = (output_grad - d_mu) / std - (matmul_result - mu) * d_input_mu / power(std, 3)
        
        # Compute gradient of matmul
        
        # Gradient of A @ B with respect to A is: d_(A @ B) @ B^T
        # Gradient of A @ B with respect to B is: A^T @ d_(A @ B)
        A = node.inputs[0]
        B = node.inputs[1]
        return [matmul(layernorm_grad, transpose(B, -2, -1)),
                matmul(transpose(A, -2, -1), layernorm_grad)]


class MatMulSoftmaxOp(Op):
    """Fused matrix multiplication and softmax operation."""

    def __call__(
        self, 
        node_A: Node, 
        node_B: Node, 
        dim: int = -1
    ) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={
                "dim": dim
            },
            name=f"MatMulSoftmax({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and softmax result."""
        assert len(input_values) == 2
        
        # Compute matmul
        
        matmul_result = None
        if input_values[0].ndim == 2:
            # Compute 2D matrix multiplication
            A = input_values[0]
            B = input_values[1]
            assert A.shape[1] == B.shape[0]
            matmul_result = A @ B
        else:
            # Compute 3D matrix multiplication, a series of 2D matrix multiplications (often a 2D matmul for each element in batch)
            results = []
            for i in range(len(input_values[0])):
                A = input_values[0][i]
                if input_values[1].ndim == 2:
                    B = input_values[1]
                else:
                    B = input_values[1][i]
                assert A.shape[1] == B.shape[0]
                results.append(A @ B)
            matmul_result = torch.stack(results, dim=0)
        
        # Compute softmax
        
        # Calculate max value along a specified dimension
        max_values = matmul_result.max(dim=node.dim, keepdim=True)[0]

        # Calculate e^(x - max(x))
        exp_values = torch.exp(matmul_result - max_values)
        
        # Normalize these results to be between 0 and 1 and sum to 1
        return exp_values / exp_values.sum(dim=node.dim, keepdim=True)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        # First compute the forward pass result we need for softmax gradient
    
        # Compute gradient of softmax
    
        # Get matmul result
        matmul_result = matmul(node.inputs[0], node.inputs[1])
        
        # Get the softmax of the input
        sftmx = softmax(matmul_result, node.dim)
        
        # Sum the products of the output gradient and each softmax term to calculate the gradient of the sum
        sum_grad = sum_op(output_grad * sftmx, dim=node.dim, keepdim=True)
        
        # Return overall gradient of softmax with respect to the input
        softmax_grad = sftmx * (output_grad - sum_grad)
        
        # Compute gradient of matmul
        
        # Gradient of A @ B with respect to A is: d_(A @ B) @ B^T
        # Gradient of A @ B with respect to B is: A^T @ d_(A @ B)
        A = node.inputs[0]
        B = node.inputs[1]
        return [matmul(softmax_grad, transpose(B, -2, -1)),
                matmul(transpose(A, -2, -1), softmax_grad)]

# Create global instances of the fused ops
matmul_layernorm = MatMulLayerNormOp()
matmul_softmax = MatMulSoftmaxOp()