#!/usr/bin/env bash
set -u
cd ~/Code/HN/HyperPEER
PY=~/Code/HN/peer-adaptive-k/.venv/bin/python
T=/home/bard/Code/HN/peer-adaptive-k/checkpoints/wt_k256_long.pt
D=/home/bard/Code/HN/peer-adaptive-k/data_wikitext
ANNEAL=results/stage3_t1_control_wt2anneal/ckpt_best.pt
DRV=logs_var_ln_driver.log
log(){ echo "$(date "+%F %T") $*" | tee -a "$DRV"; }
log "VAR DRIVER START (pid $$)"
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "$used" -lt 1500 ]; then log "GPU free (${used} MiB)"; break; fi
  log "GPU busy (${used} MiB), waiting..."; sleep 60
done
for beta in 0.01 0.1 1.0; do
  tag=$(echo $beta | tr "." "p")
  log "learned_var beta=$beta start"
  $PY -m experiments.run_learned_noise --arm learned_var --init-from "$ANNEAL" \
      --teacher-ckpt "$T" --data-dir "$D" --steps 3000 --lr 6e-5 --beta $beta \
      --eval-every 500 --eval-batches 24 --batch 2 --ctx 256 \
      --no-wandb --run-suffix "_b${tag}" > "logs_lnvar_b${tag}.log" 2>&1
  log "learned_var beta=$beta done (rc=$?)"
done
log "VAR DRIVER DONE"
