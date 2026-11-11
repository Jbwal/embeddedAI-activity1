#include <stdio.h>
#include <stdlib.h>

#define NB_PIXELS 48
#define MAX_EPOQUES 1000
#define SEED 42

/* =====================================================
   Structure d'un motif
   ===================================================== */
typedef struct
{
    float x[NB_PIXELS];
    int d;
} Motif;

/* =====================================================
   Lecture d'un motif

   . -> 0
   * -> 1
   ===================================================== */
int lire_motif(const char *nom_fichier,Motif *m)
{
    FILE *f = fopen(nom_fichier, "r");

    if (f == NULL)
    {
        printf("Impossible d'ouvrir %s\n",nom_fichier);
        return 0;
    }

    int idx = 0;
    char c;

    while (idx < NB_PIXELS && (c = fgetc(f)) != EOF)
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

    if (idx != NB_PIXELS)
    {
        printf("Motif incomplet dans %s\n",nom_fichier);
        fclose(f);
        return 0;
    }

    if (fscanf(f, "%d", &(m->d)) != 1)
    {
        printf("Erreur lecture classe\n");
        fclose(f);
        return 0;
    }

    fclose(f);
    return 1;
}

/* =====================================================
   Score réel

   a = somme(wj*xj) + b
   ===================================================== */
float calcul_score(const float x[NB_PIXELS],const float w[NB_PIXELS],float b)
{
    float a = b;

    for (int j = 0; j < NB_PIXELS; j++)
    {
        a += w[j] * x[j];
    }

    return a;
}

/* =====================================================
   Décision

   Étape 2 :
   seuil = 0.5
   ===================================================== */
int decision(float a)
{
    return (a >= 0.5f) ? 1 : 0;
}

/* =====================================================
   Widrow-Hoff

   delta = a - d

   wj <- wj - eta*delta*xj
   b  <- b  - eta*delta
   ===================================================== */
void mise_a_jour_widrow_hoff(const float x[NB_PIXELS],float w[NB_PIXELS],float *b,float a,int d,float eta)
{
    float delta = a - d;

    for (int j = 0; j < NB_PIXELS; j++)
    {
        w[j] -= eta * delta * x[j];
    }

    *b -= eta * delta;
}

/* =====================================================
   Perte quadratique

   L = (1/(2N))*Σ(d-a)^2

   Ici N = 2
   ===================================================== */
float calcul_perte(const Motif *zero,const Motif *un,const float w[NB_PIXELS],float b)
{
    float a0 = calcul_score(zero->x, w, b);

    float a1 = calcul_score(un->x,w,b);

    float L = ((zero->d - a0) * (zero->d - a0) + (un->d - a1) * (un->d - a1) ) / 4.0f;

    return L;
}

/* =====================================================
   Affichage visuel des poids
   ===================================================== */
void afficher_carte_poids(const float w[NB_PIXELS])
{
    printf("\n=== CARTE DES POIDS ===\n");

    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 6; j++)
        {
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
   Sauvegarde de la perte
   ===================================================== */
void sauver_perte_csv(float pertes[],int nb_epoques)
{
    FILE *f = fopen("perte_epoques.csv", "w");

    if (f == NULL)
        return;

    fprintf(f,"epoque,perte\n");

    for (int i = 0; i < nb_epoques; i++)
    {
        fprintf(f,"%d,%f\n",i + 1,pertes[i]);
    }

    fclose(f);
}

/* =====================================================
   Sauvegarde des poids
   ===================================================== */
void sauver_poids_csv(
    const float w[NB_PIXELS])
{
    FILE *f = fopen("poids.csv", "w");

    if (f == NULL)
        return;

    fprintf(f,"pixel,poids\n");

    for (int i = 0; i < NB_PIXELS; i++)
    {
        fprintf(f,"%d,%f\n",i,w[i]);
    }

    fclose(f);
}

/* =====================================================
   Sauvegarde paramètres
   ===================================================== */
void sauver_parametres(float b,float eta,int epoques)
{
    FILE *f = fopen("parametres.txt","w");

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
void copier_motif(const Motif *src,Motif *dst)
{
    for(int i = 0 ; i < NB_PIXELS ; i++)
    {
        dst->x[i] = src->x[i];
    }

    dst->d = src->d;
}

/* =====================================================
   Ajout de bruit

   Inverse exactement n pixels distincts
   ===================================================== */
void ajouter_bruit(Motif *m,int n)
{
    int deja_inverse[NB_PIXELS];

    for(int i = 0 ; i < NB_PIXELS ; i++)
    {
        deja_inverse[i] = 0;
    }

    int compteur = 0;

    while(compteur < n)
    {
        int p = rand() % NB_PIXELS;

        if(!deja_inverse[p])
        {
            m->x[p] = 1.0f - m->x[p];

            deja_inverse[p] = 1;

            compteur++;
        }
    }
}

/* =====================================================
   Test de robustesse pour un pourcentage de bruit
   ===================================================== */
float taux_erreur_bruit(const Motif *original, const float w[NB_PIXELS], float b, int pourcentage_bruit)
{
    int erreurs = 0;

    int nb_inversions = (int)(0.5f + NB_PIXELS * pourcentage_bruit / 100.0f);

    for(int essai = 0 ; essai < 50 ; essai++)
    {
        Motif copie;

        copier_motif(original,&copie);

        ajouter_bruit(&copie,nb_inversions);

        float score = calcul_score(copie.x,w,b);

        int prediction = decision(score);

        if(prediction != copie.d)
        {
            erreurs++;
        }
    }

    return 100.0f * erreurs / 50.0f;
}

/* =====================================================
   Génération du CSV bruit
   ===================================================== */
void sauver_bruit_csv(const Motif *zero, const Motif *un,const float w[NB_PIXELS],float b)
{
    FILE *f = fopen("bruit.csv","w");

    if(f == NULL)
        return;

    fprintf(f,"bruit,erreur_zero,erreur_un\n");

    for(int p = 0 ; p <= 100 ; p += 10)
    {
        float err_zero = taux_erreur_bruit(zero,w,b,p);

        float err_un = taux_erreur_bruit(un,w,b,p);

        fprintf(f,"%d,%.2f,%.2f\n",p,err_zero,err_un);
    }

    fclose(f);
}

/* =====================================================
   Programme principal
   ===================================================== */
int main()
{
    Motif zero;
    Motif un;

    if (!lire_motif("../patterns/zero.txt",&zero) || !lire_motif("../patterns/un.txt",&un))
    {
        return 1;
    }

    /* mêmes poids initiaux que l'étape 1 */
    srand(SEED);

    float w[NB_PIXELS];

    for (int j = 0; j < NB_PIXELS; j++)
    {
        w[j] = -0.1f + ((float)rand() / RAND_MAX) * 0.2f;
    }

    float b = -0.5f;
    float eta = 0.01f;

    float perte = 1.0f;
    int epoque = 0;

    float historique_perte[MAX_EPOQUES];

    while (perte > 1e-4 && epoque < MAX_EPOQUES)
    {
        /* ordre mélangé */
        if (rand() % 2)
        {
            float a0 = calcul_score(zero.x,w,b);

            mise_a_jour_widrow_hoff(zero.x,w,&b,a0,zero.d,eta);

            float a1 = calcul_score(un.x,w,b);

            mise_a_jour_widrow_hoff(un.x,w,&b,a1,un.d,eta);
        }
        else
        {
            float a1 = calcul_score(un.x,w,b);

            mise_a_jour_widrow_hoff(un.x,w,&b,a1,un.d,eta);

            float a0 = calcul_score(zero.x,w,b);

            mise_a_jour_widrow_hoff(zero.x,w,&b,a0,zero.d,eta);
        }

        /* perte calculée avec
           poids figés */
        perte = calcul_perte(&zero,&un,w,b);

        historique_perte[epoque] = perte;

        epoque++;
    }

    printf("\n=== APPRENTISSAGE WIDROW-HOFF ===\n");

    printf("Nombre d'epoques : %d\n",epoque);

    printf("Perte finale : %.8f\n",perte);

    printf("Biais final : %.6f\n",b);

    /* scores finaux */
    float score_zero = calcul_score(zero.x,w,b);

    float score_un = calcul_score(un.x,w,b);

    printf("\nMotif ZERO :\n");
    printf("Score = %.6f\n",score_zero);
    printf("Prediction = %d\n",decision(score_zero));

    printf("\nMotif UN :\n");
    printf("Score = %.6f\n",score_un);
    printf("Prediction = %d\n",decision(score_un));

    afficher_carte_poids(w);

    sauver_perte_csv(historique_perte,epoque);

    sauver_poids_csv(w);

    sauver_parametres(b,eta,epoque);

    printf("\nFichiers generes :\n");
    printf("- perte_epoques.csv\n");
    printf("- poids.csv\n");
    printf("- parametres.txt\n");
    
    printf("\n========== TESTS DE BRUIT ==========\n");

    int niveaux[] = {10,30,50};

    for(int i = 0 ; i < 3 ; i++){
        int p = niveaux[i];

        float err_zero = taux_erreur_bruit(&zero,w,b,p);

        float err_un = taux_erreur_bruit(&un,w,b,p);

        printf("\nBruit %d %%\n",p);

        printf("Zero : %.2f %% d'erreurs\n",err_zero);

        printf("Un : %.2f %% d'erreurs\n",err_un);
    }

    /* test libre */

    Motif mon_test;

    if (lire_motif("../patterns/mon_chiffre.txt",&mon_test))
    {
        float score = calcul_score(mon_test.x,w,b);

        printf("\nmon_chiffre.txt\n");

        printf("Score = %.6f\n",score);

        printf("Prediction = %d\n",decision(score));
    }

    sauver_bruit_csv(&zero,&un,w,b);

    printf("- bruit.csv\n");

    return 0;
}