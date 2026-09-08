# Segmentation des scans TLG

Segmente en régions de mise en page toutes les pages du dépôt HuggingFace
[`Zual/TLG_libre_scans`](https://huggingface.co/datasets/Zual/TLG_libre_scans)
(415 lignes, **409 volumes distincts**, 22,7 Go, ~176 000 pages), en vue d'une
océrisation ultérieure. Un JSON par page dans `regions/`.

## Lancer

```bash
./lancer.sh                 # tout le dépôt, 8 workers sur 2 GPU
./lancer.sh /autre/sortie 5 # ailleurs, et seulement 5 volumes
```

Reprise sans perte : relancer repart des pages manquantes. Chaque page a son
JSON, chaque volume achevé une marque `_termine`, chaque volume non décodable
une marque `_ignore`.

## Format de sortie

```json
{"page": 30, "taille": [1240, 1755], "resolution": 150.0,
 "regions": [{"nature": "Text", "boite": [181, 251, 927, 594],
              "ordre": 1, "utile": true, "intitule": false}]}
```

`boite` est en pixels de l'image donnée au modèle, dont `taille` donne les
dimensions : pour reporter les boîtes sur une image océrisée à une autre
échelle, multiplier par `largeur_ocr / taille[0]`. Les archives d'images
portent en plus `taille_source`, dimensions de l'image d'origine.

`utile` marque les régions à océriser, `intitule` les titres et intertitres.

## Environnement

`surya-ocr` est **épinglé à 0.17.1** : à partir de la 0.20.0, surya n'exécute
plus le modèle de mise en page en local mais via un VLM servi par vLLM dans
Docker. La 0.17.1 est la dernière version à modèle torch local, et elle exige
`transformers<5`.

```bash
uv venv --python 3.11 .venv-seg
VIRTUAL_ENV=.venv-seg uv pip install "surya-ocr==0.17.1" "transformers<5" pypdfium2 pillow
```

Le rendu PDF passe par `pypdfium2` et non par poppler, absent de la machine
cible et non installable sans droits.

## Choix de conception, et ce qui les motive

Tous les chiffres sont mesurés sur 2× RTX A6000.

**4 workers par GPU.** Un seul processus ne sature pas la carte : 4,6 pages/s
à 1 processus, 5,1 à 2, 5,25 à 3, 5,8 à 4. Le rendement décroît vite, et
au-delà la mémoire ne suit plus.

**Côté long borné à 1400 px** plutôt qu'un dpi fixe. Les formats de page
varient énormément selon la source : à 150 dpi, une page Gallica sortait en
7201×9744, que surya redécoupe en tuiles au-delà de 1500 px. À régions
identiques (81 dans les deux cas sur la page témoin), le rendu est 7,6× plus
rapide et le GPU 1,4× plus rapide.

**Manifeste dédoublonné.** 415 lignes pour 409 chemins : un même PDF sert
plusieurs œuvres. Sans dédoublonnage, deux workers segmentent le même volume.

**Pool de rendu recréé à chaque volume.** Le home est sur NFS : un PDF effacé
alors qu'un processus le tient encore ouvert survit en `.nfsXXXX`, et le
corpus entier resterait sur le disque.

**Rendu déporté, téléchargement anticipé.** Le rendu part dans un pool de
processus et le volume suivant se télécharge pendant que le courant occupe le
GPU, pour que celui-ci n'attende ni le disque ni le réseau.

## Limite connue

Les 8 volumes `.djvu` du manifeste sont ignorés : ni `ddjvu` ni roue pip
autonome sur la machine cible. Ils sont marqués `_ignore`.
