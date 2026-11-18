import numpy as np


def distance_logp(dist_m, sigma):
    """Gaussian log-likelihood of the GPS-to-arc distance: GPS noise is roughly
    zero-mean Gaussian around the true road position."""
    d = np.asarray(dist_m, float)
    return -0.5 * (d / sigma) ** 2 - np.log(sigma * np.sqrt(2.0 * np.pi))


def heading_logp(heading_diff_deg, sigma_deg):
    """Gaussian penalty on the angle between GPS heading and the DIRECTED arc bearing.
    Twin arcs share geometry, so distance cannot separate them; heading can (0 vs 180).
    A neutral 90 (slow points) is the same for both twins, so it cannot mislead."""
    h = np.asarray(heading_diff_deg, float)
    return -0.5 * (h / sigma_deg) ** 2


def classifier_logp(proba, dist_m, w, sigma):
    """Classifier probability mixed with the distance term:
    w * log(clip(p, 1e-6)) + (1 - w) * distance log-likelihood. w=1 trusts the classifier
    alone, w=0 is the distance-only HMM."""
    p = np.clip(np.asarray(proba, float), 1e-6, 1.0)
    return w * np.log(p) + (1.0 - w) * distance_logp(dist_m, sigma)
