"""Gradient descent with autograd on the stopping-distance data.

Run with:  python autograd_loop.py
Then comment out the two zero_() lines and run it again.
"""
import torch

speed = torch.tensor([0.4, 0.7, 0.9, 1.0, 1.1, 1.2, 1.3, 1.3, 1.4, 1.5,
                      1.6, 1.7, 1.8, 1.8, 1.9, 2.0, 2.0, 2.3, 2.4, 2.5])
distance = torch.tensor([0.2, 2.2, 1.0, 3.4, 2.8, 2.4, 2.6, 4.6, 6.0, 2.0,
                         3.2, 3.2, 4.2, 8.4, 4.6, 4.8, 5.6, 5.4, 9.2, 8.5])

m = torch.tensor(6.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

for epoch in range(1, 501):
    loss = ((m * speed + b - distance) ** 2).mean()    # predict, measure
    loss.backward()                                    # slopes
    with torch.no_grad():                              # step
        m -= 0.1 * m.grad
        b -= 0.1 * b.grad
    m.grad.zero_()                                     # clear
    b.grad.zero_()
    if epoch <= 5 or epoch % 100 == 0:
        # This loss was measured before the epoch's update.
        print(f"{epoch:3d}  m = {m.item():.4f}  b = {b.item():.4f}  loss = {loss.item():.4f}")
