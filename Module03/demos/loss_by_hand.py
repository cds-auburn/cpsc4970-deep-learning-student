"""Try values of m and b by hand and watch the loss.

Run with:  python loss_by_hand.py
The data is the 20 stopping-distance measurements from M3-010, in tens.
"""
import torch

speed = torch.tensor([0.4, 0.7, 0.9, 1.0, 1.1, 1.2, 1.3, 1.3, 1.4, 1.5,
                      1.6, 1.7, 1.8, 1.8, 1.9, 2.0, 2.0, 2.3, 2.4, 2.5])
distance = torch.tensor([0.2, 2.2, 1.0, 3.4, 2.8, 2.4, 2.6, 4.6, 6.0, 2.0,
                         3.2, 3.2, 4.2, 8.4, 4.6, 4.8, 5.6, 5.4, 9.2, 8.5])


def model(x, m, b):
    return m * x + b


def loss_fn(prediction, target):
    return ((prediction - target) ** 2).mean()


def show(m, b):
    loss = loss_fn(model(speed, m, b), distance)
    print(f"m = {m:7.4f}   b = {b:7.4f}   loss = {loss.item():7.4f}")


show(6.0, 1.0)            # our guess
show(3.6628, -0.9784)     # what one pass of learning found

# Hold b still and change only m.  Which way does the loss go?
for m in (6.0, 5.0, 4.0, 3.0, 2.0):
    show(m, -0.9784)

# Your turn: can you find an m and b with a loss below 2.17?
show(3.5, -1.0)
