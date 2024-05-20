import deepxde as dde
import numpy as np

# Definisci il dominio e le condizioni iniziali/di contorno
geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

def heat_equation(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t - dy_xx

def boundary(x, on_boundary):
    return on_boundary


# Condizioni iniziali
def initial_func(x):
    return np.sin(np.pi * x[:, 0:1])

# Crea il modello (la rete neurale deve essere definita come in precedenza)
net = dde.nn.FNN([2] + [50] * 3 + [1], "tanh", "Glorot normal")

# Definisci i vincoli per il modello
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

# Carica il modello addestrato
model.restore("heat_equation_model_hPINN")

# Usa il modello caricato per fare predizioni
X_test = geomtime.random_points(1000)
y_test = model.predict(X_test)

# Eventuale confronto con la soluzione analitica
def analytical_solution(x):
    t = x[:, 1:2]
    return np.exp(-np.pi**2 * t) * np.sin(np.pi * x[:, 0:1])

y_true = analytical_solution(X_test)
error = np.linalg.norm(y_test - y_true, 2) / np.linalg.norm(y_true, 2)
print(f"Relative L2 error: {error}")
