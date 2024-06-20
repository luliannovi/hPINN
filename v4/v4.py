import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt

# Imposta il seed per la riproducibilità
dde.config.set_random_seed(42)
np.random.seed(42)


# Definizione della PDE, che verrà utilizzato per il calcolo dell'errore
# sulla PDE tramite auto-differenziazione automatica della rete neurale
def heat_equation(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t - dy_xx


# Soluzione analitica al problema, utilizzata per il calcolo e il
# confronto con i valori veritieri
def analytical_solution(x):
    t = x[:, 1:2]
    return np.exp(-np.pi ** 2 * t) * np.sin(np.pi * x[:, 0:1])


# Definizione del dominio e delle condizioni iniziali/di contorno
geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Condizione al contorno
bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)

# Condizione iniziale del problema
ic = dde.icbc.IC(
    geomtime,
    lambda x: np.sin(np.pi * x[:, 0:1]),
    lambda _, on_initial: on_initial,
)

# Parametri per il Penalty method
mu_F = 0.1
mu_h = 0.1
beta_F = 1.01
beta_h = 1.01
iterations = 100

# Definisci la rete neurale: feed-forward con 3 layer da 12 neuroni ciascuno
net = dde.nn.FNN([2] + [12] * 2 + [1], "tanh", "Glorot normal")


# Funzione per creare il modello con i coefficienti di penalità aggiornati
def create_model(mu_F, mu_h):
    # Definisci i vincoli con i coefficienti di penalità aggiornati
    data = dde.data.TimePDE(
        geomtime,
        heat_equation,
        [bc, ic],
        num_domain=2540,
        num_boundary=80,
        num_initial=160,
        num_test=2540
    )

    # Crea il modello
    model = dde.Model(data, net)
    # Definizione dei coefficienti di penalità per le diverse parti della loss function
    # mu_h viene applicata alle condizioni al contorno e iniziali
    # mu_F viene applicata alla loss function riguardante la PDE
    loss_weights = [mu_h, mu_h, mu_F]
    model.compile("adam", lr=1e-3, loss_weights=loss_weights)
    return model


# Lista vuota per salvare la perdita durante l'addestramento
losshistory_list = []

# Inizializzazione delle liste per i loss medi
mean_train_losses = []
mean_test_losses = []

# Loop per il metodo di penalità
for k in range(iterations):
    # Crea il modello con i coefficienti di penalità aggiornati
    model = create_model(mu_F, mu_h)

    # Addestra il modello
    losshistory, train_state = model.train(epochs=100, display_every=50)

    # Calcolo della media dei loss per ogni fase
    mean_train_loss = np.mean(np.concatenate(losshistory.loss_train))
    mean_test_loss = np.mean(np.concatenate(losshistory.loss_test))

    # Aggiunta alla lista dei loss medi
    mean_train_losses.append(mean_train_loss)
    mean_test_losses.append(mean_test_loss)

    # Incrementa i coefficienti di penalità
    mu_F *= beta_F
    mu_h *= beta_h

# Salvataggio del modello
model.save("heat_equation_model_hPINN")

# Valutazione finale dei risultati tramite soluzione analitica
X_test = geomtime.random_points(1000)
y_test = model.predict(X_test)
y_true = analytical_solution(X_test)
error = ((y_true - y_test)**2).mean(axis=1)

print(f"Final Relative L2 error: {error.mean()}")

# Salva i risultati finali per l'analisi
np.savetxt("X_test_final.csv", X_test, delimiter=",")
np.savetxt("y_test_final.csv", y_test, delimiter=",")
np.savetxt("y_true_final.csv", y_true, delimiter=",")

# Grafico 3D dei risultati finali senza distinzione delle iterazioni
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test[:, 0], X_test[:, 1], y_test, label="Predicted", s=1)
ax.set_xlabel("$x$")
ax.set_ylabel("$t$")
ax.set_zlabel("$u(x,t)$")
ax.set_title("Predicted solution of the heat equation")
ax.legend()
plt.savefig("predicted_solution_final.png")
plt.show()

# Initialize steps array
steps = np.arange(0, iterations, 1)
# Plotting mean train and test losses
plt.figure(figsize=(10, 6))
plt.plot(steps, mean_train_losses, label="Mean Train Loss", marker='o')
plt.plot(steps, mean_test_losses, label="Mean Test Loss", marker='x')
plt.xlabel("# Iterations")
plt.ylabel("Loss")
plt.yscale('log')
plt.title("Mean Train and Test Losses vs Steps")
plt.legend()
plt.grid(True)
plt.show()
