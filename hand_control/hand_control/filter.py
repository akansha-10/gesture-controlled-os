import math


def smoothing_factor(delta_t, cutoff):
    r = 2 * math.pi * cutoff * delta_t
    return r / (r + 1)


def exponential_smoothing(alpha, x, x_prev):
    return alpha * x + (1 - alpha) * x_prev


class OneEuroFilter:
    """
    One Euro Filter for smoothing noisy signals.
    Reference: https://cristal.univ-lille.fr/~casiez/1euro/
    """

    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def __call__(self, t, x):
        dt = t - self.t_prev
        if dt <= 0:
            return x

        a_d = smoothing_factor(dt, self.d_cutoff)
        dx = (x - self.x_prev) / dt
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(dt, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat
