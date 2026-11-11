import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# Chargement des données produites par le programme C
# =====================================================

erreurs = pd.read_csv("erreurs_epoques.csv")
bruit = pd.read_csv("bruit.csv")

# =====================================================
# Création d'une figure à deux panneaux
# =====================================================

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

# =====================================================
# Panneau 1 : erreurs selon l'époque
# =====================================================

axes[0].plot(erreurs["epoque"], erreurs["erreurs"], marker='o', linewidth=2, color='blue')

axes[0].set_title("Erreurs selon l'époque")

axes[0].set_xlabel("Époque")

axes[0].set_ylabel("Nombre d'erreurs")

axes[0].grid(True)

# =====================================================
# Panneau 2 : taux d'erreur selon le bruit
# =====================================================

axes[1].plot(bruit["bruit"],bruit["erreur_zero"],marker='o',linewidth=2,label='Chiffre 0')

axes[1].plot(bruit["bruit"],bruit["erreur_un"],marker='s',linewidth=2,label='Chiffre 1')

axes[1].set_title("Taux d'erreur selon le bruit")

axes[1].set_xlabel("Bruit (%)")

axes[1].set_ylabel("Taux d'erreur (%)")

axes[1].set_xlim(0, 100)
axes[1].set_ylim(0, 100)

axes[1].legend()
axes[1].grid(True)

# =====================================================
# Mise en page et sauvegarde
# =====================================================

plt.tight_layout()

plt.savefig("resultats_perceptron.png",dpi=300,bbox_inches="tight")

plt.show()