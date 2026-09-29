#!/bin/bash
# Segmenter tout le dépôt en occupant les deux A6000.
#
# Sur les deux A6000, débit total mesuré selon le nombre de workers :
# 8 -> 10,5 pages/s, 12 -> 11,4, 16 -> 10,9. L'optimum est donc 6 par carte ;
# au-delà, la contention l'emporte et la mémoire approche la saturation
# (38 Go sur 48 à 16 workers).
#
# Les workers se servent dans une file commune (réservation atomique par
# volume) plutôt que dans une tranche fixe : les volumes vont de 0,2 Mo à
# 700 Mo, un découpage statique laisserait un worker finir seul.
#
#   ./lancer.sh [sortie] [plafond_volumes]

set -u
ICI=/people/pommeret/jean-claude
SORTIE=${1:-$ICI/regions}
PLAFOND=${2:-0}
PAR_GPU=${PAR_GPU:-6}
GPUS=${GPUS:-2}
WORKERS=$((GPUS * PAR_GPU))
LOGS=$ICI/.travail/logs
mkdir -p "$LOGS" "$SORTIE"

# Une réservation laissée par un processus mort bloquerait son volume : la
# campagne démarre donc sur une table nette.
find "$SORTIE" -name _encours -delete 2>/dev/null

# CC pointe vers un compilateur conda disparu : triton compile au chargement.
export CC=/usr/bin/gcc CXX=/usr/bin/g++
# Rien dans /tmp, qui est partagé entre tous les comptes de la machine.
export TRITON_CACHE_DIR=$ICI/.travail/cache-triton
export TORCHINDUCTOR_CACHE_DIR=$ICI/.travail/cache-inductor
export TOKENIZERS_PARALLELISM=false
# Chaque worker a déjà son pool de rendu : laisser torch prendre 96 cœurs par
# processus les ferait se piétiner.
export OMP_NUM_THREADS=4
# Le rendu des manuscrits coûte 0,375 s/page contre 0,092 pour l'imprimé :
# trois processus de rendu par worker, sinon le GPU attend le CPU.
export RENDUS=${RENDUS:-3}
# Les manuscrits font des pages lourdes : sans segments extensibles, la
# fragmentation du cache CUDA provoque des dépassements sur les gros lots.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# L'inventaire du dépôt se construit une fois : huit workers interrogeant
# l'API de front se font refuser (HTTP 429).
export INVENTAIRE=$ICI/.travail/inventaire.json
if [ ! -s "$INVENTAIRE" ]; then
  echo "construction de l'inventaire..."
  "$ICI/.venv-seg/bin/python" -c "
import sys; sys.path.insert(0, '$ICI')
import segmente
v = segmente.inventaire()
print(f'{len(v)} volumes, {sum(o for _, o in v)/1e9:.1f} Go')" || exit 1
fi

echo "$WORKERS workers ($PAR_GPU par GPU sur $GPUS), sortie $SORTIE"
for ((p = 0; p < WORKERS; p++)); do
  CUDA_VISIBLE_DEVICES=$((p % GPUS)) PART=$p \
    "$ICI/.venv-seg/bin/python" "$ICI/segmente.py" "$SORTIE" "$PLAFOND" \
    > "$LOGS/part$p.log" 2>&1 &
  echo "  part $p -> GPU $((p % GPUS))  (pid $!)"
done

echo "logs : $LOGS/part*.log"
echo "avancement : find $SORTIE -name '*.json' | wc -l"
wait
echo "=== tous les workers ont fini ==="
