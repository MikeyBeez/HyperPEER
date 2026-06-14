#!/usr/bin/env bash
set -u
cd ~/Code/HN/HyperPEER
PY=~/Code/HN/peer-adaptive-k/.venv/bin/python
T=/home/bard/Code/HN/peer-adaptive-k/checkpoints/wt_k256_long.pt
D=/home/bard/Code/HN/peer-adaptive-k/data_wikitext
ANNEAL=results/stage3_t1_control_wt2anneal/ckpt_best.pt
DRV=logs_overnight_ln_driver.log
log(){ echo "$(date "+%F %T") $*" | tee -a "$DRV"; }
log "DRIVER START (pid $$)"
# 0) own the GPU: wait until free
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "$used" -lt 1500 ]; then log "GPU free (${used} MiB)"; break; fi
  log "GPU busy (${used} MiB), waiting..."; sleep 60
done
# 1) wt2 recursion verdict (skip if already computed)
if [ ! -s eval_wt2_all.out ]; then
  log "computing wt2 verdict -> eval_wt2_all.out"
  {
    echo "=== WT2 RECURSION VERDICT  (precise: ctx512, 200x2 seqs, teacher wt_k256_long) ==="
    echo "pre-registered: recursion (rec) wins iff rec_CE <= ctl_CE - 0.010 nats"
    echo "--- ctl  (t1_control, T=1) ---"
    $PY -m experiments.eval_student --ckpt results/stage3_t1_control_wt2ctl/ckpt_best.pt \
        --t-steps 1 --teacher-ckpt "$T" --data-dir "$D" --batches 200 --batch-size 2 --ctx 512
    echo "--- rec  (rederive, T=2 step) ---"
    $PY -m experiments.eval_student --ckpt results/stage3_rederive_wt2rec/ckpt_best.pt \
        --t-steps 2 --rederive step --teacher-ckpt "$T" --data-dir "$D" --batches 200 --batch-size 2 --ctx 512
  } > eval_wt2_all.out 2>&1
  log "verdict eval done"
else
  log "eval_wt2_all.out exists, skipping verdict"
fi
# 2) learned-noise probe: 4 arms, matched 3000-step budget from the converged anneal ckpt
for arm in t1_control rederive iso_step learned_dir; do
  log "PROBE arm=$arm start"
  $PY -m experiments.run_learned_noise --arm "$arm" --init-from "$ANNEAL" \
      --teacher-ckpt "$T" --data-dir "$D" --steps 3000 --lr 6e-5 \
      --eval-every 500 --eval-batches 24 --batch 2 --ctx 256 \
      --no-wandb --run-suffix _ln > "logs_ln_${arm}.log" 2>&1
  log "PROBE arm=$arm done (rc=$?)"
done
log "DRIVER DONE"
