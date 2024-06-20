import matplotlib.pyplot as plt
import numpy as np
# Carica i dati di test salvati
X_test = np.loadtxt("../v4/X_test_final.csv", delimiter=",")
y_test = np.loadtxt("../v4/y_test_final.csv", delimiter=",")
y_true = np.loadtxt("../v4/y_true_final.csv", delimiter=",")

# Grafico 3D dei dati di test
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test[:, 0], X_test[:, 1], y_test, label="Predicted", s=1)
ax.scatter(X_test[:, 0], X_test[:, 1], y_true, label="True", s=1, color='orange')
ax.set_xlabel("$x$")
ax.set_ylabel("$t$")
ax.set_zlabel("$u(x,t)$")
ax.set_title("Predicted vs True Solution of the Heat Equation")
ax.legend()
plt.show()
