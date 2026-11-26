# OTP-FM student results

This inspection reads the saved 17-run batch at
`/data/lthung/flow2pc/distill/otpfm/`. As of the 2026-11-08 inspection, all 17
students completed, with zero failed/blocked/running jobs. Both evaluation
reports, selected checkpoints and all listed comparison images are present;
evaluation steps agree with the selected checkpoint steps. Arch, Sphere and
LiDAR completed after their data-path fix. No training, evaluation or test suite
was run for this inspection.

## Teacher and student population EMD

These are saved W1 EMD scores from `eval/population/evaluation.json`, evaluated
using the selected student checkpoint. CITE/Multi/EB, ST, GOM and Beijing rows
use the saved mean over held-out times; knot and tracking use observed-time
reconstruction means excluding the initial time. Arch/Sphere use the middle-time
test population, and LiDAR uses its final observed endpoint. The 5D CITE/Multi metric is
standardized; the other rows use each evaluator's recorded raw coordinates.
Compare teacher and student within a row, not numerical scales across datasets.

| Dataset / teacher recipe | Teacher EMD | Student ODE EMD | Scope |
| --- | ---: | ---: | --- |
| CITE-5D | 0.867000 | 0.877333 | Held-out time |
| CITE-50D | 43.121042 | 42.841518 | Held-out time |
| CITE-100D | 48.923553 | 48.930341 | Held-out time |
| Multi-5D | 0.912809 | 0.917671 | Held-out time |
| Multi-50D | 46.237329 | 45.892403 | Held-out time |
| Multi-100D | 52.916254 | 52.393759 | Held-out time |
| EB | 0.006095 | 0.005732 | Held-out time |
| ST | 105.832184 | 104.934566 | Held-out time |
| Knot | 0.015181 | 0.017260 | Observed-time mean |
| Cell tracking | 35.382988 | 39.559279 | Six-frame observed-time mean |
| GOM W2 | 0.128954 | 0.129700 | Held-out-time mean |
| GOM W2Inf | 0.135564 | 0.135731 | Held-out-time mean |
| Beijing W2 | 12.816486 | 12.807416 | Held-out-time mean |
| Beijing W2Inf | 16.175100 | 16.184419 | Held-out-time mean |
| Arch | 0.627394 | 0.627160 | Middle-time test |
| Sphere | 0.760534 | 0.749730 | Middle-time test |
| LiDAR | 0.103349 | 0.126474 | Final observed endpoint, raw coordinates |

## Interpretation

The students learned: every completed run reduced its saved velocity-validation
selection score substantially, and the logged scalars contain no nonfinite
values. Saved population reports contain no numerical-error entries. GOM and
Beijing retain held-out EMD within 0.6% of their teachers. Knot has a very close
visual trajectory and velocity MSE 0.006955 against teacher velocity energy
96.329727; its absolute mean EMD increase is 0.002079.

The three synthetic retries all finished 20,000 updates and selected step 20,000:

| Dataset | Initial validation velocity MSE | Selected MSE | MSE / teacher energy |
| --- | ---: | ---: | ---: |
| Arch | 3.505013 | 0.000053769 | 0.00335% |
| Sphere | 2.630211 | 0.001738627 | 0.15333% |
| LiDAR | 3.990976 | 0.009798655 | 0.56965% |

Arch is a very close teacher imitation: final paired raw-coordinate RMSE is
0.009020, and teacher/student plots almost coincide. Sphere also follows its
teacher, with final paired RMSE 0.101196 and full noninitial-time mean EMD
0.457044 teacher versus 0.459642 student (+0.57%). Its final endpoint EMD rises
from 0.153553 to 0.169553 (+10.4%), although middle-time EMD is slightly lower.

Both teachers retain the endpoint-only baseline limitation. Their plots show
poor middle-time agreement with the true Arch/Sphere populations, inherited by
the students. Close distillation does not establish recovery of the curved
benchmark path.

LiDAR preserves the broad cluster layout in the saved projection but has a
22.4% larger full-dimensional raw endpoint EMD (0.103349 to 0.126474) and final
paired RMSE 0.515788. It learned the queried velocity field, with remaining
rollout approximation error. The historical plot/report uses standardized
coordinates; the shared population score in the table uses raw coordinates.
Those two numerical scales must not be mixed.

The results do not establish uniformly faithful velocity distillation:

- CITE-50D's final validation velocity MSE is 0.010529 against teacher energy
  0.035071 (about 30% relative MSE). CITE-100D is 0.017416 / 0.038012 (about
  46%). Good held-out population EMD does not measure pointwise velocity
  agreement.
- At day 7, CITE-50D EMD is 41.798922 teacher versus 50.837474 student, and
  CITE-100D is 43.880853 versus 55.795949: student degradation of 21.6% and
  27.2%, respectively. Multi-100D also degrades by 22.2% at day 7. Their held-out
  day alone masks this later drift.
- Multi-5D has close held-out EMD, but its full noninitial-time mean is 0.671325
  teacher versus 0.779739 student, a 16.1% increase.
- Tracking's mean EMD increases by 11.8%. The saved overlay shows additional
  student shape distortion at late frames, especially frame 115. This is a
  completed run with remaining approximation error. Its selected step is 38,000,
  while training finished at 50,000. The tracking selection score 0.012785 is
  the existing worst global/late/final velocity MSE divided by teacher energy,
  rather than a raw MSE directly comparable to the other datasets.

For CITE/Multi/EB and ST, these profiles reserve the left-out population for
benchmark evaluation and select the student on reserved observed-time rows.
For GOM/Beijing, temporal holdouts are also used for velocity validation and
checkpoint selection; their EMD is a validation/tuning result. Knot/tracking
have no held-out times. These distinctions are retained in the dataset policy.
The real-data teacher sweeps themselves ranked teachers using the saved benchmark
EMD; those scores should not be described as an untouched test of teacher
selection. The synthetic teachers were selected on reserved endpoint rows;
Arch/Sphere middle-time populations stayed out of teacher/student selection.
CITE-50D's tracking status is `pending_sync`; its local model, metrics and
figures are complete.

Useful existing figures:

- [Knot teacher/student trajectories](/data/lthung/flow2pc/distill/otpfm/knot/hclt-teacher-s0/eval/comparison_trajectories.png)
- [Tracking teacher/student overlays](/data/lthung/flow2pc/distill/otpfm/cell_tracking/hclt-teacher-s0/eval/comparison_overlay.png)
- [CITE-100D marginal comparison](/data/lthung/flow2pc/distill/otpfm/cite100/hclt-teacher-s0/eval/comparison_marginals.png)
- [Arch teacher/student comparison](/data/lthung/flow2pc/distill/otpfm/arch/hclt-best-teacher-seed0/eval/comparison_marginals.png)
- [Sphere teacher/student comparison](/data/lthung/flow2pc/distill/otpfm/sphere/hclt-best-teacher-seed0/eval/comparison_marginals.png)
- [LiDAR teacher/student comparison](/data/lthung/flow2pc/distill/otpfm/lidar/hclt-best-teacher-seed0/eval/comparison_marginals.png)

## Resolved synthetic data-path failures

All three failures were `FileNotFoundError` on a prepared `manifest.json` before
the student run directory/checkpoints existed. The snapshots moved from
`/data/pqhung_group/pdtung/flow2pc/data/mfm/prepared/DATASET/seed42` to
`/data/pqhung_group/pdtung/flow2pc/data/DATASET/seed42`.

The relocated Arch, Sphere and LiDAR manifests retain the original identities;
every array hash, including the split files and LiDAR metric geometry, matches
the corresponding teacher record. Their distill presets now explicitly set
`teacher.data_root`. Restore validates the saved identity, arrays, preprocessing,
splits and time grid, permitting only this directory relocation. Teacher
artifacts and student objectives are unchanged. Launcher preflight now checks
these prepared files before starting workers.

The gitignored machine recipe is:

```bash
bash local_runs/rerun_failed_otpfm.sh
```

It preserves the full 17-run summary and skips completed students. The three
retries have now finished, so no rerun is needed. W&B remains `flow2pc`; outputs
remain in the existing OTP distillation root. The relocated datasets were
verified against their complete teacher-bound contracts after substituting only
the explicit root path. Regression cases were added but not executed.
