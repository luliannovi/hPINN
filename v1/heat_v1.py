import deepxde as dde
import numpy as np

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

# Compila e addestra il modello
model = dde.Model(data, net)
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(epochs=10000)

# Valutazione e visualizzazione dei risultati
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
