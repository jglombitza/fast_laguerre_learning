import numpy as np
import jax as jx
from scipy.special import genlaguerre
from matplotlib import pyplot as plt
import keras
from keras.models import Sequential
import jax.numpy as jnp

lay = keras.layers


def lg_mode(p, l, r, theta):
    """
    Compute simplified Laguerre-Gaussian mode (no z-dependence).

    Parameters
    ----------
    p : int
        Radial mode index (non-negative integer).
    l : int
        Azimuthal mode index (integer, can be negative).
    r : np.ndarray
        Radial coordinate grid (normalized units).
    theta : np.ndarray
        Azimuthal angle grid in radians.

    Returns
    -------
    np.ndarray
        Complex field amplitude at each point in the (r, theta) grid.
    """
    l_abs = abs(l)
    L = genlaguerre(p, l_abs)(2 * r**2)
    amp = (r**l_abs) * np.exp(-(r**2)) * L
    phase = np.exp(1j * l * theta)
    return amp * phase


def synthesize_phase(coeffs):
    """
    Synthesize phase pattern from Laguerre-Gaussian mode coefficients.

    Parameters
    ----------
    coeffs : np.ndarray
        Complex coefficients for each mode in MODES.

    Returns
    -------
    np.ndarray
        Phase pattern in radians, computed from the superposition of LG modes.

    Notes
    -----
    Uses global variables r, theta (coordinate grids) and MODES (mode list).
    """
    r, theta = make_grid(32)

    E = np.zeros_like(r, dtype=np.complex128)
    for c, (p, l) in zip(coeffs, MODES):
        E += c * lg_mode(p, l, r, theta)
    return np.angle(E)


def lg_mode_indices(max_order=4):
    """
    Generate list of (p, l) mode indices up to a maximum order.

    Parameters
    ----------
    max_order : int, optional
        Maximum mode order where 2*p + |l| <= max_order (default is 4).

    Returns
    -------
    list of tuple
        List of (p, l) tuples representing valid Laguerre-Gaussian mode indices.
    """
    modes = []
    for p in range(max_order + 1):
        for l in range(-max_order, max_order + 1):
            if 2 * p + abs(l) <= max_order:
                modes.append((p, l))
    return modes


MODES = lg_mode_indices(4)
N_MODES = len(MODES)
print("Number of modes:", N_MODES)


def make_grid(N=32):
    """
    Create polar coordinate grids for LG mode computation.

    Parameters
    ----------
    N : int, optional
        Grid resolution (NxN points, default is 32).

    Returns
    -------
    r : np.ndarray
        Radial coordinate grid of shape (N, N).
    theta : np.ndarray
        Azimuthal angle grid in radians of shape (N, N).

    Notes
    -----
    Grid spans from -2 to 2 in both x and y directions.
    """
    x = np.linspace(-2, 2, N)
    y = np.linspace(-2, 2, N)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    return r, theta


def generate_dataset(N_samples=5000, n_res=32):
    """
    Generate dataset of phase images and corresponding LG coefficients.

    Parameters
    ----------
    N_samples : int, optional
        Number of samples to generate (default is 5000).
    n_res : int, optional
        Spatial resolution of output images (default is 32).

    Returns
    -------
    X : np.ndarray
        Phase images of shape (N_samples, n_res, n_res, 1) in radians.
    Y : np.ndarray
        Flattened complex coefficients of shape (N_samples, 2*N_MODES),
        where real and imaginary parts are interleaved.

    Notes
    -----
    Coefficients are normalized to unit total power (sum of |c|^2 = 1).
    Uses global coordinate grids r, theta for phase synthesis.
    """
    X = np.zeros((N_samples, n_res, n_res, 1))
    Y = np.zeros((N_samples, 2 * N_MODES))

    for i in range(N_samples):
        coeffs = (np.random.randn(N_MODES) + 1j * np.random.randn(N_MODES)) * 0.3

        # --- normalize coefficients to unit total power ---
        coeffs /= np.sqrt(np.sum(np.abs(coeffs) ** 2))

        phase = synthesize_phase(coeffs)

        X[i, ..., 0] = phase
        Y[i, 0::2] = coeffs.real
        Y[i, 1::2] = coeffs.imag

    return X, Y


X_train, Y_train = generate_dataset(20000)
X_val, Y_val = generate_dataset(1000)


fig, axes = plt.subplots(4, 4, figsize=(14, 12))
axes = axes.ravel()
# --- Plot 16 examples ---


def plot_image(arr, coeffs, ax, fig):
    """
    Plot phase image with legend showing dominant mode contributions.

    Parameters
    ----------
    arr : np.ndarray
        2D array of phase values to display.
    coeffs : np.ndarray
        Complex LG mode coefficients.
    ax : matplotlib.axes.Axes
        Axes object to plot on.
    fig : matplotlib.figure.Figure
        Figure object for adding colorbar.

    Notes
    -----
    Displays the top 4 modes by power fraction in the legend.
    Uses global variable 'i' for title (should be passed as parameter).
    """
    power = np.abs(coeffs) ** 2
    frac = power / power.sum()

    top = np.argsort(frac)[-4:][::-1]
    legend_entries = [f"(p,l)={MODES[k]}: {frac[k]:.2f}" for k in top]

    im = ax.imshow(arr)
    ax.set_title(f"X_train example {i}")
    ax.legend(legend_entries, loc="upper right", frameon=True)

    fig.colorbar(im, ax=ax, shrink=0.85, label="phase [rad]")


for i in range(16):
    ax = axes[i]
    phase = X_train[i, ..., 0]
    coeffs = Y_train[i, 0::2] + 1j * Y_train[i, 1::2]
    plot_image(phase, coeffs, ax, fig)

plt.tight_layout()
plt.show()
plt.close("all")


# --- Custom JAX-based Layer for L2 normalization ---
class JAXL2Norm(lay.Layer):
    """
    Custom Keras layer for L2 normalization using JAX.

    Parameters
    ----------
    axis : int, optional
        Axis along which to compute the norm (default is -1).
    epsilon : float, optional
        Small constant to avoid division by zero (default is 1e-8).
    """

    def __init__(self, axis=-1, epsilon=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon

    def call(self, inputs):
        """
        Apply L2 normalization to inputs.

        Parameters
        ----------
        inputs : array_like
            Input tensor to normalize.

        Returns
        -------
        array_like
            L2-normalized tensor with the same shape as inputs.
        """
        norm = jnp.linalg.norm(inputs, axis=self.axis, keepdims=True)
        # Normalize, avoid division by zero
        return inputs / (norm + self.epsilon)

    def get_config(self):
        """
        Get layer configuration for serialization.

        Returns
        -------
        dict
            Configuration dictionary containing axis and epsilon.
        """
        config = super().get_config()
        config.update({"axis": self.axis, "epsilon": self.epsilon})
        return config


model = Sequential(
    [
        lay.Input(shape=(32, 32, 1)),
        lay.Conv2D(64, 3, activation="elu", padding="same"),
        lay.Conv2D(64, 3, activation="elu", padding="same"),
        lay.AvgPool2D((2, 2)),
        lay.Conv2D(128, 3, activation="elu", padding="same"),
        lay.Conv2D(128, 3, activation="elu", padding="same"),
        lay.AvgPool2D((2, 2)),
        lay.Conv2D(128, 3, activation="elu", padding="same"),
        lay.GlobalAveragePooling2D(),
        lay.Dense(128, activation="elu"),
        lay.Dropout(0.3),
        lay.Dense(2 * N_MODES),
        JAXL2Norm(),
    ]
)

model.compile(
    optimizer="adam",
    loss="mse",
)

model.summary()


model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=10, batch_size=64)


# pick one validation example
coeffs_pred = model.predict(X_val[:])
# coeffs_pred = Y_val


for i in range(5):
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    axes = axes.ravel()
    phase = X_val[i, ..., 0]
    coeffs = Y_val[i, 0::2] + 1j * Y_val[i, 1::2]
    plot_image(phase, coeffs, axes[0], fig)
    axes[0].set_title("input image")

    c_pred = coeffs_pred[i, 0::2] + 1j * coeffs_pred[i, 1::2]
    phase_reco = synthesize_phase(c_pred)
    plot_image(phase_reco, c_pred, axes[1], fig)
    axes[1].set_title("predicted image")
    plt.tight_layout()


plt.show()
plt.close("all")
