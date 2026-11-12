#include <stdio.h>
#include <stdlib.h>

#define NB_PIXELS 48
#define NB_CLASSES 10
#define NB_MOTIFS 10
#define NB_VARIANTES 50
#define MAX_EPOQUES 1000
#define SEED 42

#define ETA 0.01f
#define SEUIL 0.0001f
#define NB_PIXELS_BRUIT 5

/* =====================================================
   Structure d'un motif
   ===================================================== */
typedef struct
{
    float x[NB_PIXELS];
    int d;
} Motif;

/* =====================================================
   Poids et biais
   ===================================================== */
float W[NB_CLASSES][NB_PIXELS];
float b[NB_CLASSES];

/* =====================================================
   Lecture d'un motif

   . -> 0
   * -> 1
   ===================================================== */
int lire_motif(const char *nom_fichier, Motif *m)
{
    FILE *f = fopen(nom_fichier, "r");

    if (f == NULL)
    {
        printf("ERREUR : impossible d'ouvrir %s\n", nom_fichier);
        return 0;
    }

    int idx = 0;
    char c;

    while ((c = fgetc(f)) != EOF && idx < NB_PIXELS)
    {
        if (c == '.')
        {
            m->x[idx++] = 0.0f;
        }
        else if (c == '*')
        {
            m->x[idx++] = 1.0f;
        }
    }

    printf("Pixels lus : %d / %d\n", idx, NB_PIXELS);

    if (idx != NB_PIXELS)
    {
        printf("ERREUR : motif incomplet dans %s\n", nom_fichier);
        fclose(f);
        return 0;
    }

    if (fscanf(f, "%d", &m->d) != 1)
    {
        printf("ERREUR : lecture de la classe impossible dans %s\n",
               nom_fichier);

        fclose(f);
        return 0;
    }

    printf("Classe lue : %d\n", m->d);

    fclose(f);

    return 1;
}


/* =====================================================
   Création du vecteur cible

   Exemple pour le chiffre 2 :
   [0,0,1,0,0,0,0,0,0,0]
   ===================================================== */
void creer_cible(int chiffre, float d[NB_CLASSES])
{
    for (int i = 0; i < NB_CLASSES; i++){
        d[i] = 0.0f;
    }
    d[chiffre] = 1.0f;
}

/* =====================================================
   Initialisation des poids et des biais

   Poids entre -0.1 et 0.1
   Biais à zéro
   ===================================================== */
void initialiser()
{
    for (int i = 0; i < NB_CLASSES; i++)
    {
        for (int j = 0; j < NB_PIXELS; j++)
        {
            W[i][j] = -0.1f
                    + ((float)rand() / RAND_MAX) * 0.2f;
        }

        b[i] = 0.0f;
    }
}

/* =====================================================
   Calcul des dix scores

   ai = somme(Wij*xj) + bi
   ===================================================== */
void calcul_scores(const float x[NB_PIXELS],float a[NB_CLASSES])
{
    for (int i = 0; i < NB_CLASSES; i++)
    {
        a[i] = b[i];

        for (int j = 0; j < NB_PIXELS; j++)
        {
            a[i] += W[i][j] * x[j];
        }
    }
}

/* =====================================================
   Prédiction par argmax

   En cas d'égalité, le plus petit indice est conservé.
   ===================================================== */
int decision(const float a[NB_CLASSES])
{
    int indice_max = 0;

    for (int i = 1; i < NB_CLASSES; i++)
    {
        if (a[i] > a[indice_max])
            indice_max = i;
    }

    return indice_max;
}

/* =====================================================
   Apprentissage Widrow-Hoff

   delta[i] = a[i] - d[i]

   Wij <- Wij - eta*delta[i]*x[j]
   bi  <- bi  - eta*delta[i]

   Les dix erreurs sont calculées avant les mises à jour.
   ===================================================== */
void mise_a_jour_widrow_hoff(const float x[NB_PIXELS],int chiffre,float eta)
{
    float a[NB_CLASSES];
    float d[NB_CLASSES];
    float delta[NB_CLASSES];

    creer_cible(chiffre, d);

    /* Calcul des dix scores */
    calcul_scores(x, a);

    /* Calcul des dix erreurs avant mise à jour */
    for (int i = 0; i < NB_CLASSES; i++)
    {
        delta[i] = a[i] - d[i];
    }

    /* Mise à jour des poids et des biais */
    for (int i = 0; i < NB_CLASSES; i++)
    {
        for (int j = 0; j < NB_PIXELS; j++)
        {
            W[i][j] -= eta * delta[i] * x[j];
        }

        b[i] -= eta * delta[i];
    }
}

/* =====================================================
   Calcul de la perte

   L = (1/(2N))*somme_k somme_i (dki-aki)^2

   N = 10

   Aucun apprentissage pendant cette étape.
   ===================================================== */
float calcul_perte(Motif motifs[NB_MOTIFS])
{
    float somme = 0.0f;

    float a[NB_CLASSES];
    float d[NB_CLASSES];

    for (int k = 0; k < NB_MOTIFS; k++)
    {
        creer_cible(motifs[k].d, d);

        calcul_scores(motifs[k].x, a);

        for (int i = 0; i < NB_CLASSES; i++)
        {
            float erreur = d[i] - a[i];

            somme += erreur * erreur;
        }
    }

    return somme / (2.0f * NB_MOTIFS);
}

/* =====================================================
   Évaluation des dix motifs propres

   Retourne le nombre de classes correctes.
   Aucun apprentissage.
   ===================================================== */
int evaluer(Motif motifs[NB_MOTIFS])
{
    int corrects = 0;

    float a[NB_CLASSES];

    for (int k = 0; k < NB_MOTIFS; k++)
    {
        calcul_scores(motifs[k].x, a);

        int prediction = decision(a);

        if (prediction == motifs[k].d)
            corrects++;
    }

    return corrects;
}

/* =====================================================
   Mélange des dix motifs
   ===================================================== */
void melanger(int ordre[NB_MOTIFS])
{
    for (int i = 0; i < NB_MOTIFS; i++)
        ordre[i] = i;

    for (int i = NB_MOTIFS - 1; i > 0; i--)
    {
        int j = rand() % (i + 1);

        int temp = ordre[i];
        ordre[i] = ordre[j];
        ordre[j] = temp;
    }
}

/* =====================================================
   Copie d'un motif
   ===================================================== */
void copier_motif(const Motif *src, Motif *dst)
{
    for (int i = 0; i < NB_PIXELS; i++)
        dst->x[i] = src->x[i];

    dst->d = src->d;
}

/* =====================================================
   Ajout de bruit

   Inverse exactement n pixels distincts.
   ===================================================== */
void ajouter_bruit(Motif *m, int n)
{
    int deja_inverse[NB_PIXELS] = {0};

    int compteur = 0;

    while (compteur < n)
    {
        int p = rand() % NB_PIXELS;

        if (!deja_inverse[p])
        {
            m->x[p] = 1.0f - m->x[p];

            deja_inverse[p] = 1;

            compteur++;
        }
    }
}

/* =====================================================
   Sauvegarde de l'historique d'apprentissage

   Taux de reconnaissance et perte par époque.
   ===================================================== */
void sauver_historique_csv(float pertes[],float taux[],int nb_epoques)
{
    FILE *f = fopen("historique.csv", "w");

    if (f == NULL)
    {
        printf("Erreur creation historique.csv\n");
        return;
    }

    fprintf(f, "epoque,taux_reconnaissance,perte\n");

    for (int i = 0; i < nb_epoques; i++)
    {
        fprintf(f, "%d,%.2f,%.8f\n",i + 1, taux[i], pertes[i]);
    }

    fclose(f);
}

/* =====================================================
   Sauvegarde des prédictions sur motifs propres
   ===================================================== */
void sauver_predictions_csv(Motif motifs[NB_MOTIFS])
{
    FILE *f = fopen("predictions.csv", "w");

    if (f == NULL)
        return;

    fprintf(f, "chiffre_reel,chiffre_predit,resultat\n");

    float a[NB_CLASSES];

    for (int k = 0; k < NB_MOTIFS; k++)
    {
        calcul_scores(motifs[k].x, a);

        int prediction = decision(a);

        fprintf(f, "%d,%d,%s\n",motifs[k].d,prediction,prediction == motifs[k].d ? "Correct" : "Incorrect");
    }

    fclose(f);
}

/* =====================================================
   Sauvegarde des poids et biais
   ===================================================== */
void sauver_parametres(void)
{
    FILE *f = fopen("poids.csv", "w");

    if (f == NULL)
        return;

    fprintf(f, "sortie,pixel,poids\n");

    for (int i = 0; i < NB_CLASSES; i++)
    {
        for (int j = 0; j < NB_PIXELS; j++)
        {
            fprintf(f, "%d,%d,%.6f\n",i, j, W[i][j]);
        }
    }

    fclose(f);

    f = fopen("biais.csv", "w");

    if (f == NULL)
        return;

    fprintf(f, "sortie,biais\n");

    for (int i = 0; i < NB_CLASSES; i++)
    {
        fprintf(f, "%d,%.6f\n", i, b[i]);
    }

    fclose(f);
}

/* =====================================================
   Test de robustesse

   50 variantes par chiffre.
   10 % de bruit = 5 pixels inversés.

   Génère :
   - confusion.csv
   - taux_global.txt
   ===================================================== */
void tester_bruit(Motif motifs[NB_MOTIFS])
{
    int confusion[NB_CLASSES][NB_CLASSES] = {0};

    int total_corrects = 0;

    int nb_inversions = NB_PIXELS_BRUIT;

    for (int k = 0; k < NB_MOTIFS; k++)
    {
        for (int essai = 0; essai < NB_VARIANTES; essai++)
        {
            Motif copie;

            float a[NB_CLASSES];

            copier_motif(&motifs[k], &copie);

            ajouter_bruit(&copie, nb_inversions);

            calcul_scores(copie.x, a);

            int prediction = decision(a);

            confusion[copie.d][prediction]++;

            if (prediction == copie.d)
                total_corrects++;
        }
    }

    /* Affichage de la matrice */
    printf("\n===== MATRICE DE CONFUSION =====\n\n");

    printf("Reel/Pred");

    for (int i = 0; i < NB_CLASSES; i++)
        printf("%5d", i);

    printf("\n");

    for (int i = 0; i < NB_CLASSES; i++)
    {
        printf("%5d", i);

        for (int j = 0; j < NB_CLASSES; j++)
            printf("%5d", confusion[i][j]);

        printf("\n");
    }

    /* Sauvegarde CSV */
    FILE *f = fopen("confusion.csv", "w");

    if (f != NULL)
    {
        fprintf(f, "Reel/Pred");

        for (int i = 0; i < NB_CLASSES; i++)
            fprintf(f, ",%d", i);

        fprintf(f, "\n");

        for (int i = 0; i < NB_CLASSES; i++)
        {
            fprintf(f, "%d", i);

            for (int j = 0; j < NB_CLASSES; j++)
                fprintf(f, ",%d", confusion[i][j]);

            fprintf(f, "\n");
        }

        fclose(f);
    }

    /* Calcul du taux global */
    float taux_global = 100.0f * total_corrects / (NB_MOTIFS * NB_VARIANTES);

    printf("\n===== TAUX GLOBAL =====\n");

    printf("Bonnes reponses : %d / 500\n", total_corrects);

    printf("Taux global : %.2f %%\n", taux_global);

    /* Sauvegarde du taux global */
    f = fopen("taux_global.txt", "w");

    if (f != NULL)
    {
        fprintf(f, "Bonnes reponses : %d / 500\n",
                total_corrects);

        fprintf(f, "Taux global : %.2f %%\n",
                taux_global);

        fclose(f);
    }
}

/* =====================================================
   Programme principal
   ===================================================== */
int main()
{
    
    Motif motifs[NB_MOTIFS];

    const char *fichiers[NB_MOTIFS] =
    {
        "../patterns/zero.txt",
        "../patterns/un.txt",
        "../patterns/deux.txt",
        "../patterns/trois.txt",
        "../patterns/quatre.txt",
        "../patterns/cinq.txt",
        "../patterns/six.txt",
        "../patterns/sept.txt",
        "../patterns/huit.txt",
        "../patterns/neuf.txt"
    };

    printf("Lecture des motifs...\n");
    fflush(stdout);


    /* Lecture des dix motifs */
    for (int i = 0; i < NB_MOTIFS; i++)
    {
        
        if (!lire_motif(fichiers[i], &motifs[i]))
            return 1;

        if (motifs[i].d != i)
        {
            printf("Classe incorrecte dans %s\n", fichiers[i]);
            return 1;
        }
    }


    /* Initialisation */
    srand(SEED);

    initialiser();

    float pertes[MAX_EPOQUES];
    float taux[MAX_EPOQUES];

    int ordre[NB_MOTIFS];

    int corrects = 0;
    int epoque = 0;

    float perte = 1.0f;

    /* =================================================
       APPRENTISSAGE
       ================================================= */
    while (perte > SEUIL && epoque < MAX_EPOQUES)
    {
        /* Mélange des dix motifs */
        melanger(ordre);

        /* Apprentissage */
        for (int k = 0; k < NB_MOTIFS; k++)
        {
            int indice = ordre[k];

            mise_a_jour_widrow_hoff(
                motifs[indice].x,
                motifs[indice].d,
                ETA
            );
        }

        /* Évaluation sans apprentissage */
        perte = calcul_perte(motifs);

        corrects = evaluer(motifs);

        pertes[epoque] = perte;

        taux[epoque] = 100.0f * corrects / NB_MOTIFS;

        epoque++;

        printf("Epoque %d : %d/10 corrects | "
               "Taux : %.2f %% | Perte : %.8f\n",
               epoque,
               corrects,
               taux[epoque - 1],
               perte);
    }

    /* =================================================
       RESULTATS FINAUX
       ================================================= */
    printf("\n===== RESULTATS APPRENTISSAGE =====\n");

    printf("Nombre d'epoques : %d\n", epoque);

    printf("Classes correctes : %d/10\n", corrects);

    printf("Taux de reconnaissance : %.2f %%\n", taux[epoque - 1]);

    printf("Perte finale : %.8f\n", perte);

    if (perte <= SEUIL)
        printf("Seuil atteint.\n");
    else
        printf("Limite de 1000 epoques atteinte.\n");

    /* =================================================
       PREDICTIONS SUR MOTIFS PROPRES
       ================================================= */
    printf("\n===== PREDICTIONS SUR MOTIFS PROPRES =====\n");

    printf("Reel | Predit | Resultat\n");

    for (int k = 0; k < NB_MOTIFS; k++)
    {
        float a[NB_CLASSES];

        calcul_scores(motifs[k].x, a);

        int prediction = decision(a);

        printf("%4d | %6d | %s\n", motifs[k].d, prediction, prediction == motifs[k].d ? "Correct" : "Incorrect");
    }

    /* =================================================
       SAUVEGARDES
       ================================================= */
    sauver_historique_csv(pertes, taux, epoque);

    sauver_predictions_csv(motifs);

    sauver_parametres();

    /* =================================================
       TEST DES 500 VARIANTES
       ================================================= */
    tester_bruit(motifs);

    printf("\n===== FICHIERS GENERES =====\n");

    printf("- historique.csv\n");
    printf("- predictions.csv\n");
    printf("- confusion.csv\n");
    printf("- taux_global.txt\n");
    printf("- poids.csv\n");
    printf("- biais.csv\n");

    return 0;
}
