import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np

#Chargement des données


historique = pd.read_csv("historique.csv")

confusion = pd.read_csv("confusion.csv", index_col=0)

predictions = pd.read_csv("predictions.csv")


#Calcul du taux global


total = confusion.values.sum()

bonnes_reponses = np.trace(confusion.values)

taux_global = 100 * bonnes_reponses / total

print("\n========== RESULTATS GLOBAUX ==========")

print("Nombre total d'essais :", total)

print("Bonnes réponses :", bonnes_reponses)

print("Taux global : {:.2f} %".format(taux_global))

#Vérification des 500 variantes

assert total == 500, "Erreur : il faut 500 essais."

assert all(confusion.sum(axis=1) == 50), "Erreur : chaque chiffre doit avoir 50 essais."


#Figure à deux panneaux


fig, ax = plt.subplots(1, 2, figsize=(14, 6))


#PANNEAU 1 Taux de reconnaissance selon l'époque

ax[0].plot( historique["epoque"], historique["taux_reconnaissance"], color="blue", linewidth=2, label="Widrow-Hoff" )

ax[0].set_title( "Taux de reconnaissance sur les 10 motifs" )

ax[0].set_xlabel( "Époque" )

ax[0].set_ylabel( "Taux de reconnaissance (%)" )

ax[0].set_ylim(0, 105)

ax[0].grid(True)

ax[0].legend()


# PANNEAU 2 Matrice de confusion des 500 variantes


im = ax[1].imshow( confusion.values, cmap="Blues" )

ax[1].set_title( "Matrice de confusion (500 variantes bruitées)" )

ax[1].set_xlabel( "Chiffre prédit" )

ax[1].set_ylabel( "Chiffre réel" )

ax[1].set_xticks(range(10))

ax[1].set_yticks(range(10))

# Affichage des valeurs dans chaque case

for i in range(10):
    for j in range(10):

        couleur = "white" if confusion.iloc[i, j] > 25 else "black"

        ax[1].text(j, i,str(confusion.iloc[i, j]),ha="center",va="center",color=couleur)

fig.colorbar( im, ax=ax[1], label="Nombre de prédictions" )

# Sauvegarde de la figure


plt.tight_layout()

plt.savefig( "figure_resultats.png", dpi=300, bbox_inches="tight" )

plt.show()

#TABLEAU DES 10 PREDICTIONS

print("\n========== PREDICTIONS SUR MOTIFS PROPRES ==========\n")

print( predictions.to_string(index=False) )

#Calcul du taux sur les motifs propres

corrects_propres = ( predictions["chiffre_reel"] == predictions["chiffre_predit"] ).sum()

taux_propre = 100 * corrects_propres / 10

print("\nTaux de reconnaissance sur les motifs propres : " "{:.2f} %".format(taux_propre))

#Sauvegarde du tableau

predictions.to_csv( "tableau_predictions.csv", index=False )


#TABLEAU VISUEL DES PREDICTIONS


fig, ax = plt.subplots(figsize=(7, 5))

ax.axis("off")

table = ax.table( cellText=predictions.values, colLabels=predictions.columns, loc="center", cellLoc="center" )

table.auto_set_font_size(False)

table.set_fontsize(11)

table.scale(1, 1.6)

ax.set_title( "Prédictions sur les dix motifs propres", fontsize=14, pad=20 )

plt.savefig( "tableau_predictions.png", dpi=300, bbox_inches="tight" )

plt.show()


#RECHERCHE DES CHIFFRES CONFONDUS


matrice = confusion.values.copy()

#Suppression de la diagonale
#La diagonale représente les bonnes prédictions.

np.fill_diagonal(matrice, 0)

#Recherche de la confusion la plus fréquente

if matrice.max() > 0:
    i, j = np.unravel_index(np.argmax(matrice),matrice.shape)

    nb_confusions = matrice[i, j]

    phrase_confusion = (f"Les chiffres les plus fréquemment confondus sont "f"{i} et {j} : le chiffre réel {i} est prédit "f"{nb_confusions} fois comme étant {j}.")

else:

    phrase_confusion = ("Aucune confusion n'a été observée sur les 500 variantes bruitées.")


#LES QUATRE PHRASES D'INTERPRETATION


phrases = [(f"Le réseau reconnaît correctement "f"{taux_propre:.2f} % des dix motifs propres."),

phrase_confusion,(f"Sur les 500 variantes bruitées, le taux global "f"de reconnaissance est de {taux_global:.2f} %."),

(
    "Les dix motifs appris ne représentent pas toutes "
    "les façons possibles d'écrire les chiffres. "
    "Le réseau peut donc rencontrer des difficultés "
    "face à des écritures différentes de celles "
    "utilisées pendant l'apprentissage."
)


]


#AFFICHAGE DE L'INTERPRETATION


print("\n========== INTERPRETATION DES RESULTATS ==========\n")

for numero, phrase in enumerate(phrases, 1):

    print(str(numero) + ". " + phrase)


#SAUVEGARDE DE L'INTERPRETATION


with open( "interpretation.txt", "w", encoding="utf-8" ) as f:

    for numero, phrase in enumerate(phrases, 1):

        f.write(str(numero) + ". " + phrase + "\n")


#FIN


print("\n========== FICHIERS GENERES ==========")

print("- figure_resultats.png")

print("- tableau_predictions.png")

print("- tableau_predictions.csv")

print("- interpretation.txt")

print("\nAnalyse terminee.")