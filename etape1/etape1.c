#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NB_PIXELS 48
#define MAX_EPOQUES 1000
#define NB_TESTS_BRUIT 50
#define SEED 42

/* =====================================================
   Structure d'un motif
   ===================================================== */
typedef struct {
    float x[NB_PIXELS];
    int d;
} Motif;

/* =====================================================
   Lecture d'un motif depuis un fichier texte
   ===================================================== */
int lire_motif(const char *nom_fichier, Motif *m)
{
    FILE *f = fopen(nom_fichier, "r");

    if (f == NULL) {
        printf("Impossible d'ouvrir %s\n",
               nom_fichier);
        return 0;
    }

    int idx = 0;
    char c;

    while (idx < NB_PIXELS && (c = fgetc(f)) != EOF){
        if (c == '.') {
            m->x[idx++] = 0.0f;
        }
        else if (c == '*') {
            m->x[idx++] = 1.0f;
        }
    }

    if (idx != NB_PIXELS) {
        printf("Motif incomplet dans %s\n",
               nom_fichier);
        fclose(f);
        return 0;
    }

    if (fscanf(f, "%d", &(m->d)) != 1) {
        printf("Erreur lecture classe\n");
        fclose(f);
        return 0;
    }

    fclose(f);
    return 1;
}

/* =====================================================
   Calcul du score du neurone

   a = somme(wj*xj) + b
   ===================================================== */
float calcul_score(
    const float x[NB_PIXELS],
    const float w[NB_PIXELS],
    float b)
{
    float a = b;

    for (int j = 0; j < NB_PIXELS; j++) {
        a += w[j] * x[j];
    }

    return a;
}

/* =====================================================
   Fonction de décision
   ===================================================== */
int decision(float a)
{
    return (a >= 0.0f) ? 1 : 0;
}

/* =====================================================
   Mise à jour des poids
   ===================================================== */
void mise_a_jour(
    const float x[NB_PIXELS],
    float w[NB_PIXELS],
    float *b,
    int d,
    int y,
    float eta)
{
    int e = d - y;

    if (e != 0) {
        for (int j = 0; j < NB_PIXELS; j++) {
            w[j] += eta * e * x[j];
        }

        *b += eta * e;
    }
}

/* =====================================================
   Affichage visuel des poids appris
   ===================================================== */
void afficher_carte_poids(const float w[NB_PIXELS]) {
    printf("\n=== CARTE DES POIDS ===\n");

    for (int i = 0; i < 8; i++) {

        for (int j = 0; j < 6; j++) {

            int idx = i * 6 + j;

            if (w[idx] > 0.05f)
                printf(" + ");

            else if (w[idx] < -0.05f)
                printf(" - ");

            else
                printf(" . ");
        }

        printf("\n");
    }
}

/* =====================================================
   Sauvegarde des erreurs d'apprentissage
   ===================================================== */
void sauver_erreurs_csv(int erreurs[], int nb_epoques)
{
    FILE *f = fopen("erreurs_epoques.csv", "w");

    if (f == NULL)
        return;

    fprintf(f, "epoque,erreurs\n");

    for (int i = 0; i < nb_epoques; i++) {
        fprintf(f,"%d,%d\n",i + 1,erreurs[i]);
    }

    fclose(f);
}

/* =====================================================
   Sauvegarde des poids
   ===================================================== */
void sauver_poids_csv(const float w[NB_PIXELS]){
    FILE *f = fopen("poids.csv", "w");

    if (f == NULL)
        return;

    fprintf(f, "pixel,poids\n");
    for (int i = 0; i < NB_PIXELS; i++) {
        fprintf(f,"%d,%f\n",i,w[i]);
    }
    fclose(f);
}

/* =====================================================
   Sauvegarde des paramètres
   ===================================================== */
void sauver_parametres(float b, float eta, int epoques)
{
    FILE *f = fopen("parametres.txt", "w");

    if (f == NULL)
        return;

    fprintf(f,"SEED=%d\n",SEED);

    fprintf(f,"ETA=%f\n",eta);

    fprintf(f,"BIAIS_FINAL=%f\n",b);

    fprintf(f,"EPOQUES=%d\n",epoques);

    fclose(f);
}

/* =====================================================
   Copie d'un motif
   ===================================================== */
void copier_motif(const Motif *source,Motif *destination){
    memcpy(destination,source,sizeof(Motif));
}

/* =====================================================
   Inversion de n pixels distincts
   ===================================================== */
void ajouter_bruit(Motif *m,int nb_pixels_a_inverser){
    int deja_inverse[NB_PIXELS];
    for (int i = 0; i < NB_PIXELS; i++) {
        deja_inverse[i] = 0;
    }
    int compteur = 0;
    while (compteur < nb_pixels_a_inverser) {
        int p = rand() % NB_PIXELS;
        if (!deja_inverse[p]) {
            m->x[p] = 1.0f - m->x[p];
            deja_inverse[p] = 1;
            compteur++;
        }
    }
}

/* =====================================================
   Expériences avec bruit
   ===================================================== */
void tester_bruit(const Motif *zero, const Motif *un, const float w[NB_PIXELS], float b){
    FILE *f = fopen("bruit.csv", "w");

    if (f == NULL)
        return;

    fprintf(f,"bruit,erreur_zero,erreur_un\n");

    for (int p = 0; p <= 100; p += 10) {
        int erreurs_zero = 0;
        int erreurs_un = 0;
        int inversions = (int)(0.5 + p * NB_PIXELS / 100.0);
        /* 50 essais sur le zéro */
        for (int essai = 0; essai < NB_TESTS_BRUIT; essai++)
        {
            Motif copie;
            copier_motif(zero, &copie);
            ajouter_bruit(&copie, inversions);
            int prediction = decision(calcul_score(copie.x, w, b));
            if (prediction != 0)
                erreurs_zero++;
        }

        /* 50 essais sur le un */

        for (int essai = 0; essai < NB_TESTS_BRUIT; essai++)
        {
            Motif copie;
            copier_motif(un, &copie);
            ajouter_bruit(&copie, inversions);
            int prediction = decision(calcul_score(copie.x, w, b));

            if (prediction != 1)
                erreurs_un++;
        }

        double taux_zero = 100.0 * erreurs_zero / NB_TESTS_BRUIT;

        double taux_un = 100.0 * erreurs_un / NB_TESTS_BRUIT;

        fprintf(f, "%d,%.2f,%.2f\n", p, taux_zero, taux_un);
    }
    fclose(f);
}

/* =====================================================
   PROGRAMME PRINCIPAL
   ===================================================== */
int main(){
    Motif zero;
    Motif un;

    if (!lire_motif("../patterns/zero.txt",&zero) || !lire_motif("../patterns/un.txt",&un))
    {
        return 1;
    }

    srand(SEED); // Graine enregistrée
    float w[NB_PIXELS];

    for (int j = 0; j < NB_PIXELS; j++) {
        w[j] = -0.1f + ((float)rand() / RAND_MAX) * 0.2f;
    }

    float b = -0.5f;
    float eta = 0.01f;
    int erreurs = 1;
    int epoque = 0;

    int historique_erreurs[MAX_EPOQUES];

    while (erreurs > 0 && epoque < MAX_EPOQUES)
    {
        /* ordre mélangé */
        if (rand() % 2) {
            int y0 = decision(calcul_score(zero.x,w,b));
            mise_a_jour(zero.x,w,&b,zero.d,y0,eta);

            int y1 = decision(calcul_score(un.x,w,b));

            mise_a_jour(un.x,w,&b,un.d,y1,eta);
        }
        else {
            int y1 = decision(calcul_score(un.x,w,b));

            mise_a_jour(un.x,w,&b,un.d,y1,eta);

            int y0 = decision(calcul_score(zero.x,w,b));

            mise_a_jour(zero.x,w,&b,zero.d,y0,eta);
        }

        int err0 = (decision(calcul_score(zero.x,w,b))!= 0);

        int err1 = (decision(calcul_score(un.x,w,b))!= 1);

        erreurs = err0 + err1;

        historique_erreurs[epoque] = erreurs;

        epoque++;
    }

    printf("\nApprentissage termine en %d epoques\n",epoque);

    printf("Biais final : %.4f\n", b);

    afficher_carte_poids(w);

    sauver_erreurs_csv(historique_erreurs,epoque);

    sauver_poids_csv(w);

    sauver_parametres(b,eta,epoque);

    tester_bruit(&zero,&un,w,b);

    printf(
        "\nFichiers generes :\n"
        "- erreurs_epoques.csv\n"
        "- bruit.csv\n"
        "- poids.csv\n"
        "- parametres.txt\n");

    return 0;
}