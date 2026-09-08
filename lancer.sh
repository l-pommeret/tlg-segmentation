#!/bin/bash
# Segmenter tout le dépôt en occupant les deux A6000.
#
# Mesuré sur une A6000 : 1 processus = 4,6 pages/s, 2 = 5,1, 3 = 5,25, 4 = 5,8.
# Un processus ne sature donc pas la carte, mais le rendement décroît vite ;
# 4 par carte est le point où le gain paie encore la mémoire (4,8 Go par
# processus à LOT=32, sur 48 Go).
#
#   ./lancer.sh [sortie] [plafond_volumes]

set -u
ICI=/people/pommeret/jean-claude
SORTIE=${1:-$ICI/regions}
PLAFOND=${2:-0}
PAR_GPU=${PAR_GPU:-4}
GPUS=${GPUS:-2}
PARTS=$((GPUS * PAR_GPU))
LOGS=$ICI/.travail/logs
mkdir -p "$LOGS" "$SORTIE"

# CC pointe vers un compilateur conda disparu : triton compile au chargement.
export CC=/usr/bin/gcc CXX=/usr/bin/g++
# Rien dans /tmp, qui est partagé entre tous les comptes de la machine.
export TRITON_CACHE_DIR=$ICI/.travail/cache-triton
export TORCHINDUCTOR_CACHE_DIR=$ICI/.travail/cache-inductor
export TOKENIZERS_PARALLELISM=false
# Chaque worker a déjà son pool de rendu : laisser torch prendre 96 cœurs par
# processus les ferait se piétiner.
export OMP_NUM_THREADS=4

echo "$PARTS workers ($PAR_GPU par GPU sur $GPUS), sortie $SORTIE"
for ((p = 0; p < PARTS; p++)); do
  CUDA_VISIBLE_DEVICES=$((p % GPUS)) PART=$p PARTS=$PARTS \
    "$ICI/.venv-seg/bin/python" "$ICI/segmente.py" "$SORTIE" "$PLAFOND" \
    > "$LOGS/part$p.log" 2>&1 &
  echo "  part $p -> GPU $((p % GPUS))  (pid $!)"
done

echo "logs : $LOGS/part*.log"
echo "avancement : find $SORTIE -name '*.json' | wc -l"
wait
echo "=== tous les workers ont fini ==="
