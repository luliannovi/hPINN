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


# Parametri per il metodo di penalità
mu_F = 1.0
mu_h = 1.0
beta_F = 1.1
beta_h = 1.1
iterations = 10

# Definisci la rete neurale
net = dde.nn.FNN([2] + [50] * 3 + [1], "tanh", "Glorot normal")


# Funzione per creare il modello con i coefficienti di penalità aggiornati
def create_model(mu_F, mu_h):
    # Definisci i vincoli con i coefficienti di penalità aggiornati
    data = dde.data.TimePDE(
        geomtime,
        heat_equation,
        [dde.icbc.IC(geomtime, initial_func, boundary)],
        num_domain=4000,
        num_boundary=1000,
        num_initial=100,
    )

    # Crea il modello
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3)
    return model


# Loop per il metodo di penalità
for k in range(iterations):
    # Crea il modello con i coefficienti di penalità aggiornati
    model = create_model(mu_F, mu_h)

    # Addestra il modello
    losshistory, train_state = model.train(epochs=2000)

    # Incrementa i coefficienti di penalità
    mu_F *= beta_F
    mu_h *= beta_h

# Valutazione e visualizzazione dei risultati
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

model.save("heat_equation_model_hPINN")

X_test = geomtime.random_points(1000)
y_test = model.predict(X_test)

def analytical_solution(x):
    t = x[:, 1:2]
    return np.exp(-np.pi**2 * t) * np.sin(np.pi * x[:, 0:1])

y_true = analytical_solution(X_test)
error = np.linalg.norm(y_test - y_true, 2) / np.linalg.norm(y_true, 2)
print(f"Relative L2 error: {error}")
