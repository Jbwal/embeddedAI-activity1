#include <stdio.h>
#include <stdlib.h>

#define NB_PIXELS 48


typedef struct{
    float x[NB_PIXELS];
    int d;
} Motif;

int lire_motif(const char *nom_fichier, Motif *m){
    FILE *f = fopen(nom_fichier,"r");
    if(f==NULL){
        printf("impossible d'ouvrir le fichier %s\n", nom_fichier);
        return 0;
    }

    int idx = 0;
    char c;

    while(idx < NB_PIXELS && (c=fgetc(f)) != EOF){
        if(c=='.'){
            m->x[idx] = 0.0f;
            idx++;
        } else if (c=='*'){
            m->x[idx] = 1.0f;
            idx++;
        }
    }

    if(fscanf(f, "%d", &(m->d)) != 1){
        printf("erreur de lecture de la classe dans %s\n", nom_fichier);
        fclose(f);
        return 0;
    }
    fclose(f);
    return 1;
}

// 1. Calcul du score a
float calcul_score(const float x[NB_PIXELS], const float w[NB_PIXELS], float b) {
    float a = b; // On commence directement avec le biais
    for (int j = 0; j < NB_PIXELS; j++) {
        a += w[j] * x[j];
    }
    return a;
}

// 2. Prise de décision y_chapeau
int decision(float a) {
    return (a >= 0.0f) ? 1 : 0;
}

// 3. Mise à jour des poids et du biais
void mise_a_jour(const float x[NB_PIXELS], float w[NB_PIXELS], float *b, int d, int y_chapeau, float eta) {
    int e = d - y_chapeau; // Erreur (0 si correct, +1 ou -1 si faux)
    
    if (e != 0) { // On ne met à jour que s'il y a une erreur
        // Règle du perceptron : w_j <- w_j + eta * e * x_j
        for (int j = 0; j < NB_PIXELS; j++) {
            w[j] += eta * e * x[j];
        }
        // Biais : b <- b + eta * e
        *b += eta * e;
    }
}

void afficher_carte_poids(const float w[NB_PIXELS]) {
    printf("\n--- Carte des poids appris (8x6) ---\n");
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 6; j++) {
            int idx = i * 6 + j;
            if (w[idx] > 0.05f) {
                printf(" + "); // Pixel très favorable au 1
            } else if (w[idx] < -0.05f) {
                printf(" - "); // Pixel très favorable au 0
            } else {
                printf(" . "); // Pixel quasiment neutre
            }
        }
        printf("\n");
    }
}


int main(void) {
    Motif zero, un;
    
    // 1. Chargement des fichiers txt
    if (!lire_motif("../patterns/zero.txt", &zero) || !lire_motif("../patterns/un.txt", &un)) {
        return 1;
    }

    // 2. Initialisation des paramètres
    srand(42); // Graine enregistrée
    float w[NB_PIXELS];
    for (int j = 0; j < NB_PIXELS; j++) {
        // Tirage aléatoire dans [-0.1, 0.1]
        w[j] = -0.1f + ((float)rand() / RAND_MAX) * 0.2f;
    }
    float b = -0.5f;
    float eta = 0.01f;

    int erreurs = 1;
    int epoque = 0;

    while (erreurs > 0 && epoque < 1000) {
        // Apprentissage sur le zero
        float a0 = calcul_score(zero.x, w, b);
        int y0 = decision(a0);
        mise_a_jour(zero.x, w, &b, zero.d, y0, eta);

        // Apprentissage sur le un
        float a1 = calcul_score(un.x, w, b);
        int y1 = decision(a1);
        mise_a_jour(un.x, w, &b, un.d, y1, eta);

        // Verification sans mise à jour
        int err0 = (decision(calcul_score(zero.x, w, b)) != zero.d);
        int err1 = (decision(calcul_score(un.x, w, b)) != un.d);
        erreurs = err0 + err1;

        epoque++;
    }

    printf("Apprentissage termine en %d epoques !\n", epoque);
    afficher_carte_poids(w);

    // 3. TEST SUR TON PROPRE FICHIER PERSO
    Motif mon_test;
    if (lire_motif("../patterns/mon_chiffre.txt", &mon_test)) {
        float score_test = calcul_score(mon_test.x, w, b);
        int pred_test = decision(score_test);
        printf("\nTest sur mon_chiffre.txt :\n");
        printf("Score = %.2f -> Prediction : %d\n", score_test, pred_test);
    }

    return 0;
}