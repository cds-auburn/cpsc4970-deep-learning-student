"""Gradient descent on the stopping-distance data, with slopes from nudges.

Run with:  python gradient_descent.py
Change learning_rate and run it again: try 0.5 and 0.001.
"""
import torch

speed = torch.tensor([0.4, 0.7, 0.9, 1.0, 1.1, 1.2, 1.3, 1.3, 1.4, 1.5,
                      1.6, 1.7, 1.8, 1.8, 1.9, 2.0, 2.0, 2.3, 2.4, 2.5])
distance = torch.tensor([0.2, 2.2, 1.0, 3.4, 2.8, 2.4, 2.6, 4.6, 6.0, 2.0,
                         3.2, 3.2, 4.2, 8.4, 4.6, 4.8, 5.6, 5.4, 9.2, 8.5])


def loss_at(m, b):
    prediction = m * speed + b
    return ((prediction - distance) ** 2).mean()


def gradient(m, b, delta=0.1):
    dm = (loss_at(m + delta, b) - loss_at(m - delta, b)) / (2 * delta)
    db = (loss_at(m, b + delta) - loss_at(m, b - delta)) / (2 * delta)
    return dm, db


m, b = 6.0, 1.0
learning_rate = 0.1

print(f"  0  m = {m:.4f}  b = {b:.4f}  loss = {loss_at(m, b):.4f}")
for epoch in range(1, 501):
    dm, db = gradient(m, b)        # both slopes first...
    m = m - learning_rate * dm     # ...then both updates
    b = b - learning_rate * db
    if epoch <= 5 or epoch % 100 == 0:
        print(f"{epoch:3d}  m = {m:.4f}  b = {b:.4f}  loss = {loss_at(m, b):.4f}")
