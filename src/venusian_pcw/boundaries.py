"""Reference bow-shock model curves and manually annotated IMF event intervals."""

import numpy as np

# Published model parameters (x0, L, eccentricity); no measured crossings.
BOW_SHOCK_MODELS = {
    "低活动期": (0.626, 1.559, 1.024),
    "高活动期": (0.638, 1.687, 1.041),
}


def bow_shock_curve(x0, L, eps, theta=np.linspace(0, 2.4, 600)):
    rr = L / (1 + eps * np.cos(theta))
    return x0 + rr * np.cos(theta), rr * np.sin(theta)
