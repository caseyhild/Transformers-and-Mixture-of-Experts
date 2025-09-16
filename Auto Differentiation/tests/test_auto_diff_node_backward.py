from typing import Dict, List

import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import auto_diff as ad

def check_evaluator_output(
    evaluator: ad.Evaluator,
    input_values: Dict[ad.Node, torch.Tensor],
    expected_outputs: List[torch.Tensor],
) -> None:
    output_values = evaluator.run(input_values)
    assert len(output_values) == len(expected_outputs)
    for output_val, expected_val in zip(output_values, expected_outputs):
        print(repr(output_val))
        torch.testing.assert_close(output_val, expected_val, atol=1e-4, rtol=1e-4)
    
def test_add():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.add(x1, x2)
    y_grad = ad.Variable("y_grad")
    x1_grad, x2_grad = y.op.gradient(y, y_grad)
    evaluator = ad.Evaluator(eval_nodes=[x1_grad, x2_grad])

    check_evaluator_output(
        evaluator,
        input_values={
            x1: torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            x2: torch.tensor([[2.8, 0.7, -0.1, 0.0], [0.6, 6.6, 3.2, 3.1]]),
            y_grad: torch.tensor([[-0.4, 0.5, -5.0, 34.0], [-0.0375, 0.0, 2.32, -3.1]]),
        },
        expected_outputs=[
            torch.tensor([[-0.4, 0.5, -5.0, 34.0], [-0.0375, 0.0, 2.32, -3.1]]),
            torch.tensor([[-0.4, 0.5, -5.0, 34.0], [-0.0375, 0.0, 2.32, -3.1]]),
        ],
    )

def test_add_by_const():
    x1 = ad.Variable("x1")
    y = ad.add_by_const(x1, 2.7)
    y_grad = ad.Variable("y_grad")
    x1_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x1_grad])

    check_evaluator_output(
        evaluator,
        input_values={
            x1: torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            y_grad: torch.tensor([[-0.4, 0.5, -5.0, 34.0], [-0.0375, 0.0, 2.32, -3.1]]),
        },
        expected_outputs=[
            torch.tensor([[-0.4, 0.5, -5.0, 34.0], [-0.0375, 0.0, 2.32, -3.1]]),
        ],
    )

def test_mul():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.mul(x1, x2)
    y_grad = ad.Variable("y_grad")
    x1_grad, x2_grad = y.op.gradient(y, y_grad)
    evaluator = ad.Evaluator(eval_nodes=[x1_grad, x2_grad])

    check_evaluator_output(
        evaluator,
        input_values={
            x1: torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            x2: torch.tensor([[2.8, 0.7, -0.1, 0.0], [0.6, 6.6, 3.2, 3.1]]),
            y_grad: torch.tensor([[-0.4, 0.5, -5.0, 34.0], [-0.0375, 0.0, 2.32, -3.1]]),
        },
        expected_outputs=[
            torch.tensor([[-1.12, 0.35, 0.5, 0.0], [-0.0225, 0.0, 7.424, -9.61]]),
            torch.tensor([[0.4, 1.0, -2.5, 115.6], [-0.01125, 0.0, -13.456, -9.61]]),
        ],
    )

def test_mul_by_const():
    x1 = ad.Variable("x1")
    y = ad.mul_by_const(x1, 5.0)
    y_grad = ad.Variable("y_grad")
    x1_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x1_grad])

    check_evaluator_output(
        evaluator,
        input_values={
            x1: torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            y_grad: torch.tensor([[-0.4, 0.5, -5.0, 34.0], [-0.0375, 0.0, 2.32, -3.1]]),
        },
        expected_outputs=[
            torch.tensor([[-2.00, 2.50, -25.00, 170.00], [-0.1875, 0.0, 11.60, -15.50]]),
        ],
    )

def test_greater_than():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.greater(x1, x2)
    y_grad = ad.Variable("y_grad")
    x1_grad, x2_grad = y.op.gradient(y, y_grad)
    evaluator = ad.Evaluator(eval_nodes=[x1_grad, x2_grad])

    check_evaluator_output(
        evaluator,
        input_values={
            x1: torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            x2: torch.tensor([[2.8, 0.7, -0.1, 0.0], [0.6, 6.6, 3.2, 3.1]]),
            y_grad: torch.tensor([[-0.4, 0.5, -5.0, 34.0], [-0.0375, 0.0, 2.32, -3.1]]),
        },
        expected_outputs=[
            torch.tensor([[0.00, 0.00, 0.00, 0.00], [0.00, 0.00, 0.00, 0.00]]),
            torch.tensor([[0.00, 0.00, 0.00, 0.00], [0.00, 0.00, 0.00, 0.00]]),
        ],
    )

def test_sub():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.sub(x1, x2)
    y_grad = ad.Variable("y_grad")
    x1_grad, x2_grad = y.op.gradient(y, y_grad)
    evaluator = ad.Evaluator(eval_nodes=[x1_grad, x2_grad])

    check_evaluator_output(
        evaluator,
        input_values={
            x1: torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            x2: torch.tensor([[2.8, 0.7, -0.1, 0.0], [0.6, 6.6, 3.2, 3.1]]),
            y_grad: torch.tensor([[-0.4, 0.5, -5.0, 34.0], [-0.0375, 0.0, 2.32, -3.1]]),
        },
        expected_outputs=[
            torch.tensor([[-0.4, 0.5, -5.0, 34.0], [-0.0375, 0.0, 2.32, -3.1]]),
            torch.tensor([[0.4, -0.5, 5.0, -34.0], [0.0375, 0.0, -2.32, 3.1]]),
        ],
    )
    
def test_zeros_like():
    x1 = ad.Variable("x1")
    y = ad.zeros_like(x1)
    y_grad = ad.Variable("y_grad")
    x1_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x1_grad])

    check_evaluator_output(
        evaluator,
        input_values={
            x1: torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            y_grad: torch.tensor([[-0.4, 0.5, -5.0, 34.0], [-0.0375, 0.0, 2.32, -3.1]]),
        },
        expected_outputs=[
            torch.tensor([[0.00, 0.00, 0.00, 0.00], [0.00, 0.00, 0.00, 0.00]]),
        ],
    )
    
def test_ones_like():
    x1 = ad.Variable("x1")
    y = ad.ones_like(x1)
    y_grad = ad.Variable("y_grad")
    x1_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x1_grad])

    check_evaluator_output(
        evaluator,
        input_values={
            x1: torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            y_grad: torch.tensor([[-0.4, 0.5, -5.0, 34.0], [-0.0375, 0.0, 2.32, -3.1]]),
        },
        expected_outputs=[
            torch.tensor([[0.00, 0.00, 0.00, 0.00], [0.00, 0.00, 0.00, 0.00]]),
        ],
    )
    
def test_sum_no_keepdim():
    x1 = ad.Variable("x1")
    y = ad.sum_op(x1, dim=(0,), keepdim=False)
    y_grad = ad.Variable("y_grad")
    x1_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x1_grad])

    check_evaluator_output(
        evaluator,
        input_values={
            x1: torch.tensor([[[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]],
                              [[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]]),
            y_grad: torch.tensor([[-0.4, 0.5, -5.0, 34.0], [-0.0375, 0.0, 2.32, -3.1]]),
        },
        expected_outputs=[
            torch.tensor([[[-0.4, 0.5, -5.0, 34.0], [-0.0375, 0.0, 2.32, -3.1]],
                        [[-0.4, 0.5, -5.0, 34.0], [-0.0375, 0.0, 2.32, -3.1]]]),
        ],
    )

def test_sum_keepdim():
    x1 = ad.Variable("x1")
    y = ad.sum_op(x1, dim=(1,), keepdim=True)
    y_grad = ad.Variable("y_grad")
    x1_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x1_grad])

    check_evaluator_output(
        evaluator,
        input_values={
            x1: torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            y_grad: torch.tensor([[-0.4, 0.5, -5.0, 34.0], [-0.0375, 0.0, 2.32, -3.1]]),
        },
        expected_outputs=[
            torch.tensor([[-0.4, 0.5, -5.0, 34.0], [-0.0375, 0.0, 2.32, -3.1]]),
        ],
    )
    
def test_expand_as():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.expand_as(x1, x2)
    y_grad = ad.Variable("y_grad")
    x1_grad, x2_grad = y.op.gradient(y, y_grad)
    evaluator = ad.Evaluator(eval_nodes=[x1_grad, x2_grad])

    check_evaluator_output(
        evaluator,
        input_values={
            x1: torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            x2: torch.tensor([[[2.8, 0.7, -0.1, 0.0], [0.6, 6.6, 3.2, 3.1]],
                        [[2.8, 0.7, -0.1, 0.0], [0.6, 6.6, 3.2, 3.1]]]),
            y_grad: torch.tensor([[[-0.4, 0.5, -5.0, 34.0], [-0.0375, 0.0, 2.32, -3.1]],
                                  [[-0.4, 0.5, -5.0, 34.0], [-0.0375, 0.0, 2.32, -3.1]]]),
        },
        expected_outputs=[
            torch.tensor([[-0.8, 1.0, -10.0, 68.0], [-0.075, 0.0, 4.64, -6.2]]),
            torch.tensor([[[0.00, 0.00, 0.00, 0.00], [0.00, 0.00, 0.00, 0.00]],
                        [[0.00, 0.00, 0.00, 0.00], [0.00, 0.00, 0.00, 0.00]]]),
        ],
    )

def test_expand_as_3d():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.expand_as_3d(x1, x2, 1)
    y_grad = ad.Variable("y_grad")
    x1_grad, x2_grad = y.op.gradient(y, y_grad)
    evaluator = ad.Evaluator(eval_nodes=[x1_grad, x2_grad])

    check_evaluator_output(
        evaluator,
        input_values={
            x1: torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            x2: torch.tensor([[[2.8, 0.7, -0.1, 0.0], [0.6, 6.6, 3.2, 3.1]],
                        [[2.8, 0.7, -0.1, 0.0], [0.6, 6.6, 3.2, 3.1]]]),
            y_grad: torch.tensor([[[-0.4, 0.5, -5.0, 34.0], [-0.0375, 0.0, 2.32, -3.1]],
                                  [[-0.4, 0.5, -5.0, 34.0], [-0.0375, 0.0, 2.32, -3.1]]]),
        },
        expected_outputs=[
            torch.tensor([[-0.8, 1.0, -10.0, 68.0], [-0.075, 0.0, 4.64, -6.2]]),
            torch.tensor([[[0.00, 0.00, 0.00, 0.00], [0.00, 0.00, 0.00, 0.00]],
                        [[0.00, 0.00, 0.00, 0.00], [0.00, 0.00, 0.00, 0.00]]]),
        ],
    )
    
def test_add_dim():
    x1 = ad.Variable("x1")
    y = ad.add_dim(x1, dim=2)
    y_grad = ad.Variable("y_grad")
    x1_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x1_grad])

    check_evaluator_output(
        evaluator,
        input_values={
            x1: torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            y_grad: torch.tensor([[-0.4, 0.5, -5.0, 34.0], [-0.0375, 0.0, 2.32, -3.1]]),
        },
        expected_outputs=[
            torch.tensor([[0.00, 0.00, 0.00, 0.00], [0.00, 0.00, 0.00, 0.00]]),
        ],
    )

def test_remove_dim():
    x1 = ad.Variable("x1")
    y = ad.remove_dim(x1, dim=1)
    y_grad = ad.Variable("y_grad")
    x1_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x1_grad])

    check_evaluator_output(
        evaluator,
        input_values={
            x1: torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            y_grad: torch.tensor([[-0.4, 0.5, -5.0, 34.0], [-0.0375, 0.0, 2.32, -3.1]]),
        },
        expected_outputs=[
            torch.tensor([[0.00, 0.00, 0.00, 0.00], [0.00, 0.00, 0.00, 0.00]]),
        ],
    )
    
def test_log():
    x1 = ad.Variable("x1")
    y = ad.log(x1)
    y_grad = ad.Variable("y_grad")
    x1_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x1_grad])

    check_evaluator_output(
        evaluator,
        input_values={
            x1: torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 1.0, -5.8, 3.1]]),
            y_grad: torch.tensor([[-0.4, 0.5, -5.0, 34.0], [-0.0375, 0.0, 2.32, -3.1]]),
        },
        expected_outputs=[
            torch.tensor([[0.4, 0.25, -10.0, 10.0], [-0.125, 0.0, -0.4, -1.0]]),
        ],
    )

def test_broadcast():
    x = ad.Variable("x")
    y = ad.broadcast(x, input_shape=[3, 2], target_shape=[2, 3, 2])
    y_grad = ad.Variable("y_grad")
    x_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x_grad])

    x_val = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y_grad_val = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], 
        [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
    ])

    check_evaluator_output(
        evaluator,
        input_values={x: x_val, y_grad: y_grad_val},
        expected_outputs=[torch.tensor([[8.0, 10.0], [12.0, 14.0], [16.0, 18.0]])]
    )

def test_div():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.div(x1, x2)
    y_grad = ad.Variable("y_grad")
    x1_grad, x2_grad = y.op.gradient(y, y_grad)
    evaluator = ad.Evaluator(eval_nodes=[x1_grad, x2_grad])

    check_evaluator_output(
        evaluator,
        input_values={
            x1: torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            x2: torch.tensor([[2.5, 4.0, -0.1, 0.1], [-8.0, 5.0, -2.5, -1.0]]),
            y_grad: torch.ones((2, 4), dtype=torch.float32),
        },
        expected_outputs=[
            torch.tensor([[0.4, 0.25, -10.0, 10.0], [-0.125, 0.2, -0.4, -1.0]]),
            torch.tensor([[0.16, -0.125, -50.0, -340.0], [-0.0046875, 0, 0.928, -3.1]]),
        ],
    )

# def test_div_by_const():
#     x1 = ad.Variable("x1")
#     y = ad.div_by_const(x1, 5.0)
#     evaluator = ad.Evaluator(eval_nodes=[y])

#     check_evaluator_output(
#         evaluator,
#         input_values={x1: torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]])},
#         expected_outputs=[torch.tensor([[-0.2, 0.4, 0.1, 0.68], [0.06, 0.0, -1.16, 0.62]])],
#     )

def test_div_by_const():
    x1 = ad.Variable("x1")
    y = ad.div_by_const(x1, 5.0)
    y_grad = ad.Variable("y_grad")
    x1_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x1_grad])

    check_evaluator_output(
        evaluator,
        input_values={
            x1: torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            y_grad: torch.tensor([[-0.4, 0.5, -5.0, 34.0], [-0.0375, 0.0, 2.32, -3.1]]),
        },
        expected_outputs=[
            torch.tensor([[-0.08, 0.10, -1.00, 6.80], [-0.0075, 0.0, 0.464, -0.62]]),
        ],
    )

def test_transpose():
    x = ad.Variable("x")
    y = ad.transpose(x, 1, 0)
    y_grad = ad.Variable("y_grad")
    x_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x_grad])

    x_val = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y_grad_val = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    check_evaluator_output(
        evaluator,
        input_values={x: x_val, y_grad: y_grad_val},
        expected_outputs=[torch.tensor([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])]
    )

def test_matmul():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.matmul(x1, x2)
    y_grad = ad.Variable("y_grad")
    x1_grad, x2_grad = y.op.gradient(y, y_grad)
    evaluator = ad.Evaluator(eval_nodes=[x1_grad, x2_grad])

    x1_val = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x2_val = torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    y_grad_val = torch.ones((3, 3), dtype=torch.float32)
    x1_grad_expected = torch.tensor([[24.0, 33.0], [24.0, 33.0], [24.0, 33.0]])
    x2_grad_expected = torch.tensor([[9.0, 9.0, 9.0], [12.0, 12.0, 12.0]])

    check_evaluator_output(
        evaluator,
        input_values={
            x1: x1_val,
            x2: x2_val,
            y_grad: y_grad_val,
        },
        expected_outputs=[x1_grad_expected, x2_grad_expected],
    )

def test_matmul_3d():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.matmul(x1, x2)
    y_grad = ad.Variable("y_grad")
    x1_grad, x2_grad = y.op.gradient(y, y_grad)
    evaluator = ad.Evaluator(eval_nodes=[x1_grad, x2_grad])

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

    y_grad_val = torch.ones((2, 3, 3), dtype=torch.float32)

    x1_grad_expected = torch.tensor([[[6.0, 15.0, 24.0],
                                    [6.0, 15.0, 24.0],
                                    [6.0, 15.0, 24.0]],
                                   [[24.0, 15.0, 6.0],
                                    [24.0, 15.0, 6.0],
                                    [24.0, 15.0, 6.0]]])

    x2_grad_expected = torch.tensor([[[12.0, 12.0, 12.0],
                                    [15.0, 15.0, 15.0],
                                    [18.0, 18.0, 18.0]],
                                   [[18.0, 18.0, 18.0],
                                    [15.0, 15.0, 15.0],
                                    [12.0, 12.0, 12.0]]])

    check_evaluator_output(
        evaluator,
        input_values={
            x1: x1_val,
            x2: x2_val,
            y_grad: y_grad_val,
        },
        expected_outputs=[x1_grad_expected, x2_grad_expected],
    )

def test_softmax():
    x = ad.Variable("x")
    y = ad.softmax(x)
    y_grad = ad.Variable("y_grad")
    x_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x_grad])

    x_val = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    y_grad_val = torch.tensor([[0.5, -0.3, 0.8], [-0.2, 0.4, -0.1]], dtype=torch.float32)

    check_evaluator_output(
        evaluator,
        input_values={x: x_val, y_grad: y_grad_val},
        expected_outputs=[
            torch.tensor([
                [-0.0003, -0.1967,  0.1971],
                [-0.0192,  0.0946, -0.0754]
            ], dtype=torch.float32)
        ]
    )

def test_layernorm():
    x = ad.Variable("x")
    y = ad.layernorm(x, normalized_shape=[3])
    y_grad = ad.Variable("y_grad")
    x_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x_grad])

    x_val = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    y_grad_val = torch.tensor([[12, 4, 2], [-3, -5, 3]], dtype=torch.float32)

    check_evaluator_output(
        evaluator,
        input_values={x: x_val, y_grad: y_grad_val},
        expected_outputs=[
            torch.tensor([
                [1.2248, -2.4495,  1.2246],
                [2.0412, -4.0825, 2.0413]
            ], dtype=torch.float32)
        ]
    )

def test_relu():
    x = ad.Variable("x")
    y = ad.relu(x)
    y_grad = ad.Variable("y_grad")
    x_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x_grad])

    x_val = torch.tensor([[-1.0, 2.0, 0.0], [3.0, -4.0, 5.0]], dtype=torch.float32)
    y_grad_val = torch.ones_like(x_val)

    check_evaluator_output(
        evaluator,
        input_values={x: x_val, y_grad: y_grad_val},
        expected_outputs=[torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=torch.float32)]
    )
    
def test_sqrt():
    x = ad.Variable("x")
    y = ad.sqrt(x)
    y_grad = ad.Variable("y_grad")
    x_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x_grad])

    x_val = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]], dtype=torch.float32)
    y_grad_val = torch.ones_like(x_val)

    check_evaluator_output(
        evaluator,
        input_values={x: x_val, y_grad: y_grad_val},
        expected_outputs=[torch.tensor([[0.5, 0.353553, 0.288675], [0.288675, 0.25, 0.223607]], dtype=torch.float32)]
    )

def test_power():
    x = ad.Variable("x")
    y = ad.power(x, 3)
    y_grad = ad.Variable("y_grad")
    x_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x_grad])

    x_val = torch.tensor([[-1.0, 2.0, 0.0], [3.0, -4.0, 5.0]], dtype=torch.float32)
    y_grad_val = torch.ones_like(x_val)

    check_evaluator_output(
        evaluator,
        input_values={x: x_val, y_grad: y_grad_val},
        expected_outputs=[torch.tensor([[3.0, 12.0, 0.0], [27.0, 48.0, 75.0]], dtype=torch.float32)]
    )

def test_mean_no_keepdim():
    x1 = ad.Variable("x1")
    y = ad.mean(x1, dim=(0,), keepdim=False)
    y_grad = ad.Variable("y_grad")
    x1_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x1_grad])

    check_evaluator_output(
        evaluator,
        input_values={
            x1: torch.tensor([[[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]],
                              [[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]]),
            y_grad: torch.tensor([[-0.4, 0.5, -5.0, 34.0], [-0.0375, 0.0, 2.32, -3.1]]),
        },
        expected_outputs=[
            torch.tensor([[[-0.2, 0.25, -2.5, 17.0], [-0.01875, 0.0, 1.16, -1.55]],
                        [[-0.2, 0.25, -2.5, 17.0], [-0.01875, 0.0, 1.16, -1.55]]]),
        ],
    )

def test_mean_keepdim():
    x1 = ad.Variable("x1")
    y = ad.mean(x1, dim=(1,), keepdim=True)
    y_grad = ad.Variable("y_grad")
    x1_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x1_grad])

    check_evaluator_output(
        evaluator,
        input_values={
            x1: torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            y_grad: torch.tensor([[-0.4, 0.5, -5.0, 34.0], [-0.0375, 0.0, 2.32, -3.1]]),
        },
        expected_outputs=[
            torch.tensor([[-0.1, 0.125, -1.25, 8.5], [-0.009375, 0.0, 0.58, -0.775]]),
        ],
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
