from typing import List

import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import auto_diff as ad

def check_compute_output(
    node: ad.Node, input_values: List[torch.Tensor], expected_output: torch.Tensor
) -> None:
    output = node.op.compute(node, input_values)
    torch.testing.assert_close(output, expected_output, atol=1e-4, rtol=1e-4)
    
def test_add():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.add(x1, x2)

    check_compute_output(
        y,
        [
            torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            torch.tensor([[2.8, 0.7, -0.1, 0.0], [0.6, 6.6, 3.2, 3.1]]),
        ],
        torch.tensor([[1.80, 2.70, 0.40, 3.40], [0.90, 6.60, -2.60, 6.20]]),
    )


def test_add_by_const():
    x1 = ad.Variable("x1")
    y = ad.add_by_const(x1, 2.7)

    check_compute_output(
        y,
        [torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]])],
        torch.tensor([[1.70, 4.70, 3.20, 6.10], [3.00, 2.70, -3.10, 5.80]]),
    )

def test_mul():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.mul(x1, x2)

    check_compute_output(
        y,
        [
            torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            torch.tensor([[2.8, 0.7, -0.1, 0.0], [0.6, 6.6, 3.2, 3.1]]),
        ],
        torch.tensor([[-2.80, 1.40, -0.05, 0.00], [0.18, 0.00, -18.56, 9.61]]),
    )

def test_mul_by_const():
    x1 = ad.Variable("x1")
    y = ad.mul_by_const(x1, 2.7)

    check_compute_output(
        y,
        [torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]])],
        torch.tensor([[-2.70, 5.40, 1.35, 9.18], [0.81, 0.00, -15.66, 8.37]]),
    )

def test_greater_than():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.greater(x1, x2)

    check_compute_output(
        y,
        [
            torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            torch.tensor([[2.8, 0.7, -0.1, 0.0], [0.6, 6.6, 3.2, 3.1]]),
        ],
        torch.tensor([[0.00, 1.00, 1.00, 1.00], [0.00, 0.00, 0.00, 0.00]]),
    )

def test_sub():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.sub(x1, x2)

    check_compute_output(
        y,
        [
            torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            torch.tensor([[2.8, 0.7, -0.1, 0.0], [0.6, 6.6, 3.2, 3.1]]),
        ],
        torch.tensor([[-3.80, 1.30, 0.60, 3.40], [-0.30, -6.60, -9.00, 0.00]]),
    )
    
def test_zeros_like():
    x1 = ad.Variable("x1")
    y = ad.zeros_like(x1)

    check_compute_output(
        y,
        [torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]])],
        torch.tensor([[0.00, 0.00, 0.00, 0.00], [0.00, 0.00, 0.00, 0.00]]),
    )
    
def test_ones_like():
    x1 = ad.Variable("x1")
    y = ad.ones_like(x1)

    check_compute_output(
        y,
        [torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]])],
        torch.tensor([[1.00, 1.00, 1.00, 1.00], [1.00, 1.00, 1.00, 1.00]]),
    )

def test_sum_no_keepdim():
    x1 = ad.Variable("x1")
    y = ad.sum_op(x1, dim=(0,), keepdim=False)

    check_compute_output(
        y,
        [torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]])],
        torch.tensor([-0.70, 2.00, -5.30, 6.50]),
    )

def test_sum_keepdim():
    x1 = ad.Variable("x1")
    y = ad.sum_op(x1, dim=(1,), keepdim=True)

    check_compute_output(
        y,
        [torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]])],
        torch.tensor([[4.90], [-2.40]]),
    )

def test_expand_as():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.expand_as(x1, x2)

    check_compute_output(
        y,
        [
            torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            torch.tensor([[[2.8, 0.7, -0.1, 0.0], [0.6, 6.6, 3.2, 3.1]],
                        [[2.8, 0.7, -0.1, 0.0], [0.6, 6.6, 3.2, 3.1]]]),
        ],
        torch.tensor([[[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]],
                     [[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]]),
    )

def test_expand_as_3d():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.expand_as_3d(x1, x2, dim=(1,))

    check_compute_output(
        y,
        [
            torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            torch.tensor([[[2.8, 0.7, -0.1, 0.0], [0.6, 6.6, 3.2, 3.1]],
                        [[2.8, 0.7, -0.1, 0.0], [0.6, 6.6, 3.2, 3.1]]]),
        ],
        torch.tensor([[[-1.0, 2.0, 0.5, 3.4], [-1.0, 2.0, 0.5, 3.4]],
                     [[0.3, 0.0, -5.8, 3.1], [0.3, 0.0, -5.8, 3.1]]]),
    )
    
def test_add_dim():
    x1 = ad.Variable("x1")
    y = ad.add_dim(x1, dim=2)

    check_compute_output(
        y,
        [torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]])],
        torch.tensor([[[-1.0], [2.0], [0.5], [3.4]], [[0.3], [0.0], [-5.8], [3.1]]]),
    )
    
def test_remove_dim():
    x1 = ad.Variable("x1")
    y = ad.remove_dim(x1, dim=1)

    check_compute_output(
        y,
        [torch.tensor([[[-1.0, 2.0, 0.5, 3.4]], [[0.3, 0.0, -5.8, 3.1]]])],
        torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
    )
    
def test_log():
    x1 = ad.Variable("x1")
    y = ad.log(x1)

    check_compute_output(
        y,
        [torch.tensor([[torch.exp(torch.tensor(-3.0)), torch.exp(torch.tensor(-2.0)), torch.exp(torch.tensor(-1.0)), torch.exp(torch.tensor(0.0))], [torch.exp(torch.tensor(0.0)), torch.exp(torch.tensor(1.0)), torch.exp(torch.tensor(2.0)), torch.exp(torch.tensor(3.0))]])],
        torch.tensor([[-3.00, -2.00, -1.00, 0.00], [0.00, 1.00, 2.00, 3.00]]),
    )
    
def test_broadcast():
    x = ad.Variable("x")
    y = ad.broadcast(x, input_shape=[3, 2], target_shape=[2, 3, 2])

    check_compute_output(
        y,
        [torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])],
        torch.tensor([
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        ])
    )

def test_div():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.div(x1, x2)

    check_compute_output(
        y,
        [
            torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            torch.tensor([[2.5, 4.0, -0.1, 0.1], [-8.0, 5.0, -2.5, -1.0]]),
        ],
        torch.tensor([[-0.4, 0.5, -5.0, 34.0], [-0.0375, 0.0, 2.32, -3.1]]),
    )


def test_div_by_const():
    x1 = ad.Variable("x1")
    y = ad.div_by_const(x1, 5.0)

    check_compute_output(
        y,
        [torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]])],
        torch.tensor([[-0.2, 0.4, 0.1, 0.68], [0.06, 0.0, -1.16, 0.62]]),
    )

def test_transpose():
    x = ad.Variable("x")
    y = ad.transpose(x, 1, 0)

    check_compute_output(
        y,
        [torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])],
        torch.tensor([[[1.0, 2.0], [5.0, 6.0]], [[3.0, 4.0], [7.0, 8.0]]])
    )

def test_matmul():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.matmul(x1, x2)

    x1_val = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x2_val = torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])

    check_compute_output(
        y,
        [x1_val, x2_val],
        torch.tensor([[27.0, 30.0, 33.0], [61.0, 68.0, 75.0], [95.0, 106.0, 117.0]]),
    )

def test_matmul_3d():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.matmul(x1, x2)

    x1_val = torch.tensor([[[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0],
                           [7.0, 8.0, 9.0]],
                          [[9.0, 8.0, 7.0],
                           [6.0, 5.0, 4.0],
                           [3.0, 2.0, 1.0]]])
    
    x2_val = torch.tensor([[[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0],
                           [7.0, 8.0, 9.0]],
                          [[9.0, 8.0, 7.0],
                           [6.0, 5.0, 4.0],
                           [3.0, 2.0, 1.0]]])

    expected = torch.tensor([[[30.0, 36.0, 42.0],
                            [66.0, 81.0, 96.0],
                            [102.0, 126.0, 150.0]],
                           [[150.0, 126.0, 102.0],
                            [96.0, 81.0, 66.0],
                            [42.0, 36.0, 30.0]]])

    check_compute_output(
        y,
        [x1_val, x2_val],
        expected
    )

def test_softmax():
    x = ad.Variable("x")
    y = ad.softmax(x)

    check_compute_output(
        y,
        [torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)],
        torch.tensor([[0.0900, 0.2447, 0.6652], [0.0900, 0.2447, 0.6652]], dtype=torch.float32)
    )

def test_layernorm():
    x = ad.Variable("x")
    y = ad.layernorm(x, normalized_shape=[3])

    check_compute_output(
        y,
        [torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)],
        torch.tensor([[-1.224745, 0.0, 1.224745], [-1.224745, 0.0, 1.224745]], dtype=torch.float32)
    )


def test_relu():
    x = ad.Variable("x")
    y = ad.relu(x)

    check_compute_output(
        y,
        [torch.tensor([[-1.0, 2.0, 0.0], [3.0, -4.0, 5.0]], dtype=torch.float32)],
        torch.tensor([[0.0, 2.0, 0.0], [3.0, 0.0, 5.0]], dtype=torch.float32)
    )
    
def test_sqrt():
    x1 = ad.Variable("x1")
    y = ad.sqrt(x1)

    check_compute_output(
        y,
        [torch.tensor([[1.0, 2.0, 3.0, 4.0], [9.0, 16.0, 25.0, 36.0]])],
        torch.tensor([[1.00, 1.41421, 1.73205, 2.00], [3.00, 4.00, 5.00, 6.00]]),
    )

def test_power():
    x1 = ad.Variable("x1")
    y = ad.power(x1, 3.0)

    check_compute_output(
        y,
        [torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]])],
        torch.tensor([[-1.00, 8.00, 0.125, 39.304], [0.027, 0.00, -195.112, 29.791]]),
    )

def test_mean_no_keepdim():
    x1 = ad.Variable("x1")
    y = ad.mean(x1, dim=(0,), keepdim=False)

    check_compute_output(
        y,
        [torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]])],
        torch.tensor([-0.35, 1.00, -2.65, 3.25]),
    )

def test_mean_keepdim():
    x1 = ad.Variable("x1")
    y = ad.mean(x1, dim=(1,), keepdim=True)

    check_compute_output(
        y,
        [torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]])],
        torch.tensor([[1.225], [-0.60]]),
    )

if __name__ == "__main__":
    test_add()
    test_add_by_const()
    test_mul()
    test_mul_by_const()
    test_greater_than()
    test_sub()
    test_zeros_like()
    test_ones_like()
    test_sum_no_keepdim()
    test_sum_keepdim()
    test_expand_as()
    test_expand_as_3d()
    test_add_dim()
    test_remove_dim()
    test_log()
    test_broadcast()
    test_div()
    test_div_by_const()
    test_transpose()
    test_matmul()
    test_matmul_3d()
    test_softmax()
    test_layernorm()
    test_relu()
    test_sqrt()
    test_power()
    test_mean_no_keepdim()
    test_mean_keepdim()