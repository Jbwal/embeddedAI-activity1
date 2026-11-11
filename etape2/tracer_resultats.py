import pandas as pd
import matplotlib.pyplot as plt

# ==================================================
# Chargement des données
# ==================================================

perte = pd.read_csv("perte_epoques.csv")

perceptron = pd.read_csv("bruit_perceptron.csv")

widrow = pd.read_csv("bruit_widrow.csv")

# ==================================================
# Figure à deux panneaux
# ==================================================

fig, ax = plt.subplots(1,2,figsize=(14, 6))

# ==================================================
# PANNEAU 1
# Perte L selon l'époque
# ==================================================

ax[0].plot(perte["epoque"],perte["perte"],color="blue",linewidth=2,label="Widrow-Hoff")

ax[0].set_title("Perte L selon l'époque")

ax[0].set_xlabel(
    "Époque"
)

ax[0].set_ylabel(
    "Perte quadratique L"
)

ax[0].grid(True)

ax[0].legend()

# ==================================================
# PANNEAU 2
# Comparaison des deux méthodes
# ==================================================

# Perceptron

ax[1].plot(
    perceptron["bruit"],
    perceptron["erreur_zero"],
    "o-",
    linewidth=2,
    label="Perceptron - chiffre 0"
)

ax[1].plot(
    perceptron["bruit"],
    perceptron["erreur_un"],
    "s-",
    linewidth=2,
    label="Perceptron - chiffre 1"
)

# Widrow-Hoff

ax[1].plot(
    widrow["bruit"],
    widrow["erreur_zero"],
    "o--",
    linewidth=2,
    label="Widrow-Hoff - chiffre 0"
)

ax[1].plot(
    widrow["bruit"],
    widrow["erreur_un"],
    "s--",
    linewidth=2,
    label="Widrow-Hoff - chiffre 1"
)

ax[1].set_title(
    "Taux d'erreur selon le bruit"
)

ax[1].set_xlabel(
    "Bruit (%)"
)

ax[1].set_ylabel(
    "Taux d'erreur (%)"
)

ax[1].set_xlim(0, 100)
ax[1].set_ylim(0, 100)

ax[1].grid(True)

ax[1].legend()

# ==================================================
# Sauvegarde
# ==================================================

plt.tight_layout()

plt.savefig("figure_etape2.png",dpi=300,bbox_inches="tight")

plt.show()