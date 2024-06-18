import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
Versione con ADAM e L-BFGS
Valori di mu_f, mu_h diversi e regolatori beta
Aggiunta di fase di testing finale
"""

# Imposta il seed per la riproducibilità
dde.config.set_random_seed(42)
np.random.seed(42)


# Definizione della PDE
def heat_equation(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t - dy_xx


# Definizione del dominio e delle condizioni iniziali/di contorno
geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)


# Condizione al contorno
def boundary(x, on_boundary):
    return on_boundary


# Condizione iniziale del problema
def initial_func(x):
    return np.sin(np.pi * x[:, 0:1])


# Parametri per il Penalty method
mu_F = 0.1
mu_h = 0.1
beta_F = 1.05
beta_h = 1.05
iterations = 5

# Definisci la rete neurale: feed-forward con 5 layer da 50 neuroni ciascuno
net = dde.nn.FNN([2] + [12] * 3 + [1], "tanh", "He normal")


# Funzione per creare il modello con i coefficienti di penalità aggiornati
def create_model(mu_F, mu_h):
    # Definisci i vincoli con i coefficienti di penalità aggiornati
    data = dde.data.TimePDE(
        geomtime,
        heat_equation,
        [dde.icbc.IC(geomtime, initial_func, boundary)],
        num_domain=10000,
        num_boundary=2000,
        num_initial=200,
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
    return np.exp(-np.pi ** 2 * t) * np.sin(np.pi * x[:, 0:1])


y_true = analytical_solution(X_test)
error = np.linalg.norm(y_test - y_true, 2) / np.linalg.norm(y_true, 2)
print(f"Relative L2 error: {error}")

# Salva i risultati per l'analisi successiva
np.savetxt("X_test.csv", X_test, delimiter=",")
np.savetxt("y_test.csv", y_test, delimiter=",")
np.savetxt("y_true.csv", y_true, delimiter=",")

# Grafico 3D dei risultati
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test[:, 0], X_test[:, 1], y_test, label="Predicted", s=1)
ax.set_xlabel("$x$")
ax.set_ylabel("$t$")
ax.set_zlabel("$u(x,t)$")
ax.set_title("Predicted solution of the heat equation")
ax.legend()
plt.show()

# Grafico della perdita
plt.figure()
plt.plot(losshistory.steps, losshistory.loss_train, label="Train loss")
plt.plot(losshistory.steps, losshistory.loss_test, label="Test loss")
plt.xlabel("# Steps")
plt.ylabel("Loss")
plt.legend()
plt.show()
