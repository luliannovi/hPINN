import deepxde as dde
import numpy as np
import tensorflow as tf


# Definisci la PDE
def heat_equation(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t - dy_xx


# Definisci il dominio e le condizioni iniziali/di contorno
geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)


def boundary(x, on_boundary):
    return on_boundary


# Condizioni iniziali
def initial_func(x):
    return np.sin(np.pi * x[:, 0:1])


# Costruisci il modello PINN
data = dde.data.TimePDE(
    geomtime,
    heat_equation,
    [dde.icbc.IC(geomtime, initial_func, boundary)],
    num_domain=4000,
    num_boundary=1000,
    num_initial=100,
)

# Definisci la rete neurale
net = dde.nn.FNN([2] + [50] * 3 + [1], "tanh", "Glorot normal")

# Parametri di penalizzazione iniziali e fattori di incremento
mu_F = 1.0
mu_h = 1.0
beta_F = 1.5
beta_h = 1.5


# Funzione di perdita personalizzata
def custom_loss():
    def loss(y_true, y_pred):
        # Calcola i termini della PDE e della condizione iniziale
        error_pde = heat_equation(y_true, y_pred)
        error_ic = initial_func(y_true) - y_pred

        # Usa la funzione di perdita MSE di TensorFlow
        mse = tf.keras.losses.MeanSquaredError()
        loss_pde = mse(tf.zeros_like(error_pde), error_pde)
        loss_ic = mse(tf.zeros_like(error_ic), error_ic)

        return loss_pde + mu_F * loss_pde + mu_h * loss_ic

    return loss


# Compila e addestra il modello
model = dde.Model(data, net)
model.compile("adam", lr=1e-3, loss=custom_loss())

# Algoritmo di ottimizzazione con penalizzazione iterativa
max_iterations = 10  # Numero massimo di iterazioni esterne
tol = 1e-6  # Tolleranza per la convergenza
losshistory, train_state = None, None

for k in range(max_iterations):
    losshistory, train_state = model.train(epochs=10000, display_every=1000)
    L_Fk = np.mean(train_state.loss_train[0])
    L_hk = np.mean(train_state.loss_train[1])

    if L_Fk < tol and L_hk < tol:
        break

    # Aggiornamento dei coefficienti di penalizzazione
    mu_F *= beta_F
    mu_h *= beta_h

# Valutazione e visualizzazione dei risultati
if losshistory and train_state:
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)
