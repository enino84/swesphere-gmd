#!/bin/bash
# =====================================================================
# The whole campaign for swesphere 0.6.0, end to end.
#
#   nohup ./run_all.sh > run_all.log 2>&1 & disown
#
# Then come back and read run_all.log. Nothing else to run by hand.
#
# WHAT IT RERUNS AND WHAT IT KEEPS
#
# swesphere 0.6.0 turns the polar sponge off by default, so every
# experiment that took the default has to be measured again. Those are
# exp04, exp05, exp06, exp07, exp08, exp09 and exp11.
#
# exp01, exp02, exp03 and exp10 pass n_pole_rows explicitly at every call,
# or sweep it on purpose, so their results are unchanged by the new default
# and their CSVs are kept. Rerunning them would cost about fifteen hours and
# reproduce the same numbers.
#
# ORDER, AND WHY
#
#   stage A   exp04 in four shards, plus exp06, exp07 and exp11. Seven
#             containers, one core each, on an eight-core machine.
#   stage B   exp05, which reads what exp04 wrote and takes seconds, then
#             exp08 in eight shards. exp08 draws its ensemble from exp04's
#             records, so it cannot start earlier.
#   stage C   exp09 alone, because it measures cost per model day and a
#             timing taken while seven other containers compete is not
#             reportable. Then the figures.
# =====================================================================
set -u
cd "$(dirname "$0")"

SHARDS_A=${SHARDS_A:-4}        # shards for exp04
SHARDS_B=${SHARDS_B:-8}        # shards for exp08
export GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)

say() { echo; echo "=== $* === $(date -u +%H:%M:%S)"; }

wait_for() {   # wait_for <name prefix>
    while docker ps --format '{{.Names}}' | grep -q "^$1"; do sleep 30; done
}

run_bg() {     # run_bg <container name> <service>
    docker compose run -d --name "$1" "$2" >/dev/null \
        && echo "  lanzado $1" || echo "  FALLO al lanzar $1"
}

run_shard() {  # run_shard <name> <service> <index> <count>
    # export, not a VAR=x prefix: a prefix assignment in front of a shell
    # FUNCTION is not exported to the commands the function runs, so compose
    # would read SHARD_INDEX as unset and every shard would do shard 0.
    export SHARD_INDEX=$3 SHARD_COUNT=$4
    run_bg "$1" "$2"
    unset SHARD_INDEX SHARD_COUNT
}

say "commit $GIT_COMMIT"
docker ps --format '{{.Names}}' | grep -E '^(exp|swesphere-gmd-)' | xargs -r docker stop >/dev/null 2>&1
docker ps -aq --filter 'name=^exp' | xargs -r docker rm >/dev/null 2>&1

say "build"
docker compose build || { echo "BUILD FALLO"; exit 1; }

# The results are written by containers running as root, so they are removed
# from inside one rather than with sudo on the host.
say "borrar lo que hay que recorrer"
docker compose run --rm -T --entrypoint sh exp09 -c '
  cd /work/results || exit 0
  rm -f exp04_regimes*.csv exp05_*.csv exp06_stability*.csv \
        exp07_filter_protocol*.csv exp08_assimilation*.csv exp09_timing*.csv \
        exp11_gallery*.csv exp12_predictability*.csv \
        exp13_polar_stencil*.csv record_*.npy
  rm -rf snapshots
  echo "queda:"; ls -1 *.csv 2>/dev/null | sed "s/^/    /"
'

say "STAGE A: exp04 x$SHARDS_A, exp06, exp07, exp11, exp13"
for i in $(seq 0 $((SHARDS_A - 1))); do
    run_shard "exp04-$i" exp04 "$i" "$SHARDS_A"
done
run_bg exp06 exp06
run_bg exp07 exp07
run_bg exp11 exp11
run_bg exp13 exp13
sleep 20; docker ps --format '{{.Names}}\t{{.Status}}' | grep -E '^exp'

wait_for exp04; wait_for exp06; wait_for exp07; wait_for exp11; wait_for exp13
say "STAGE A terminada"
ls -la results/record_*.npy 2>/dev/null | wc -l | xargs echo "  registros escritos:"

say "merge de exp04"
docker compose run --rm -T --entrypoint python exp09 -c "
import pandas as pd, glob
g=sorted(glob.glob('/work/results/exp04_regimes_shard*.csv'))
if g:
    m=pd.concat([pd.read_csv(x) for x in g])
    m.to_csv('/work/results/exp04_regimes.csv', index=False)
    print('exp04:', len(m), 'registros de', len(g), 'particiones')
else:
    print('exp04: una sola particion, nada que unir')
"

say "STAGE B: exp05, despues exp08 x$SHARDS_B y exp12"
docker compose run --rm -T exp05
for i in $(seq 0 $((SHARDS_B - 1))); do
    run_shard "exp08-$i" exp08 "$i" "$SHARDS_B"
done
# exp12 reads the same records and costs minutes, so it rides along here
# instead of getting a stage of its own.
run_bg exp12 exp12
sleep 20; docker ps --format '{{.Names}}\t{{.Status}}' | grep -E '^exp'
wait_for exp08; wait_for exp12
say "STAGE B terminada"

docker compose run --rm -T --entrypoint python exp09 -c "
import pandas as pd, glob
g=sorted(glob.glob('/work/results/exp08_assimilation_shard*.csv'))
if g:
    m=pd.concat([pd.read_csv(x) for x in g])
    m.to_csv('/work/results/exp08_assimilation.csv', index=False)
    print('exp08:', len(m), 'filas de', len(g), 'particiones')
"

say "STAGE C: exp09 con la maquina vacia, despues las figuras"
docker compose run --rm -T exp09
docker compose run --rm -T gallery
docker compose run --rm -T figures

say "RESUMEN"
docker compose run --rm -T --entrypoint python exp09 -c "
import pandas as pd, glob, os
for f in sorted(glob.glob('/work/results/*.csv')):
    if 'shard' in f: continue
    try: n=len(pd.read_csv(f))
    except Exception: n='?'
    print(f'  {os.path.basename(f):42s} {n} filas')
print('  instantaneas:', len(glob.glob('/work/results/snapshots/*.npy')))
"
ls -1 figures/*.pdf 2>/dev/null | sed 's/^/  /'

say "TERMINADO $(date -u)"
echo
echo "Para el articulo, en este orden:"
echo "  results/exp04_regimes.csv        regimenes sin esponja"
echo "  results/exp05_stationarity.csv   tau y los intervalos (seccion 6.2)"
echo "  results/exp06_stability.csv      limites de dt"
echo "  results/exp07_filter_protocol.csv"
echo "  results/exp08_assimilation.csv   la seccion 7"
echo "  results/exp09_timing.csv         costo y entorno"
echo "  figures/                         las cinco de campo y las de lineas"
