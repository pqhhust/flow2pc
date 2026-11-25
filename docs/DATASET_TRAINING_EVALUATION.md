# Dataset training and evaluation reference for ALI-CFM, OTP-FM, MFM and MMFM

Reviewed: **2026-11-20**. This is a dataset-level handoff for contributors using
the same observations with different teacher methods. It covers knot, spatial
transcriptomics (ST), single-cell CITE/Multi, cell tracking, Gulf of Mexico (GOM)
and Beijing. EB is documented separately because its available representation
does not match the paper's EB benchmark. Arch/Sphere/LiDAR are outside this
document's requested scope; see [MFM data notes](MFM_LOCAL_DATA.md).

**This document does not change training, loaders, configs or checkpoints.**
It distinguishes the source experiment, the current configured experiment, and
the remaining differences between implementations. A shared dataset name or
metric name alone does not establish an identical experiment.

## 1. Sources, versions and authority

Use the source experiment's data/task definition, then the supplied data bundle
and implementation to resolve operational details. Record discrepancies rather
than silently calling an implementation choice a published setting.

| Evidence | Version/location checked |
|---|---|
| ALI-CFM paper | [arXiv 2510.01159v3](https://arxiv.org/html/2510.01159v3), sections 4.1–4.4 and appendices B–D |
| OTP-FM paper | [arXiv 2606.05327v1](https://arxiv.org/html/2606.05327v1), section 6.4 and appendices H–I |
| Supplied ALI-CFM checkout | `../ali-cfm`, HEAD `a35963ef226e1bbdc6344a944c50e0f42ed67335`; includes local reproduction changes |
| Supplied OTP-FM checkout | `../otp-fm`, HEAD `ca4fe796930414ae749eba2b299e7c7ec8cebf3d`; includes local reproduction changes |
| Flow2PC | HEAD `6e378ad4487c29112df34599e481a4d165247ad8`, plus the current uncommitted working tree |
| ALI input bundle | `/data/lthung/ali-cfm-data/README.md` and each dataset README |
| OTP input bundle | `/data/lthung/otp-fm/{gom,beijing}/README.md`, safe NPZ exports and source manifests |

HEAD identifies a checkout baseline, not a guarantee of a clean working tree.
Saved run manifests/resolved configs are authoritative for existing checkpoints;
today's YAML cannot retroactively change a trained model's experiment.

### Paper checks that matter for this handoff

- ALI-CFM: single-cell/ST interpolation benchmarks use EMD; single-cell 5D
  reporting uses whitening and inference from the preceding population.
- ALI-CFM: tracking illustrates ten training samples per frame and full-mask
  transport at six frames. The printed knot benchmark uses 1,200 marginals,
  sigma 0.1, and an unshifted middle sine branch.
- OTP-FM: GOM/Beijing report mean W2 over four held-out times; checkpoint
  selection is described using mean W2-squared. Beijing's experiment is 1D
  Dingling PM2.5, despite the larger source archive containing other variables.

These checks come from the versioned papers above. Most details below are
independently grounded in the supplied code, data READMEs and live configs.

## 2. Benchmark overview

All indices below are **zero-based** unless explicitly called frame numbers.
`tau` denotes model time. Keep physical labels and model time separately.

| Dataset/profile | State | Time grid | Times excluded from training | Population metric | Benchmark prediction source |
|---|---|---|---|---|---|
| `cite5/50/100` | First 5/50/100 saved CITE PCA coordinates | Days `[2,3,4,7]`; tau `[0,1/3,2/3,1]` | Fold 1 or 2, separately | W1/EMD; 5D in benchmark-whitened coordinates, higher D in original PCA coordinates | Preceding observed population |
| `multi5/50/100` | First 5/50/100 saved Multi PCA coordinates | Same day/model grid | Fold 1 or 2, separately | Same W1 policy | Preceding observed population |
| `st` | Aligned tumour coordinates, **(y,x)** | U2/U3/U4/U5; tau `[0,1/3,2/3,1]` | U3 or U4, separately | W1 in original aligned-coordinate units | Preceding observed slide |
| `knot` | Noisy analytic **(x,y)** | Configured analytic grid; current 120 times | None in current variant | W1/geometry diagnostics; no temporal-test score without a separate protocol | Earliest empirical or explicitly declared analytic initial law |
| `cell_tracking` | Label-4 mask pixel **(x,y)** | 115 frames; tau `i/114` | None in current setup | W1 diagnostics plus microscopy overlays | Full initial cell-4 mask |
| `gom` | 2D benchmark positions | 9 times; tau `i/8` | `[1,3,5,7]` | **W2**, raw benchmark coordinates | Earliest marginal |
| `beijing` | **1D PM2.5** in micrograms/m^3 | 13 monthly populations; tau `i/12` | `[2,5,8,11]` | **W2**, original concentration units | Earliest marginal |

The agreed main transport benchmark uses the paper's **specified target times**;
the native GOM nearby-time minimum is a separately labeled source-code parity
diagnostic. See section 7. They must not be merged into one result.

## 3. Rules shared by all four teacher methods

### 3.1 What must match, and what remains method-specific

For a controlled cross-method experiment, match the source files/hashes,
representation, time mapping, observed/withheld times, observation budget,
normalization scope, reference populations, particle-selection seed, metric
formula/units, rollout origin, scored times and aggregation. Save the actual
training row/subset indices rather than only their RNG seed.

Teacher architectures, losses, couplings, interpolants, metric-learning stages,
potential parameters and optimizers remain method-specific. A GAN or curvature
term in an ALI teacher is not a new student loss. MFM's geopath and MMFM's cubic
interpolation should not be replaced by ALI's generator to make the interface
look uniform. Pairwise MFM is not automatically a globally multi-marginal model.

Treat snapshots as **unpaired**. OT pairings and chains are inferred training
couplings, not observed trajectories. Do not treat equal row numbers across
times as particle/cell identities, including GOM and tracking mask pixels.

### 3.2 Physical versus model time

All frozen velocity adapters use `dx/dtau` in their saved input coordinates.
Day labels `[2,3,4,7]` are mapped to equally spaced ordinal pseudo-times, **not**
to `(day-2)/5`. ST uses ordered slide labels, not measured elapsed days. Tracking
uses frame indices. GOM elapsed physical units are not established by the portable
bundle. Beijing metadata stores calendar-month indices `[0,2,...,24]` separately
from tau. Any physical-time velocity conversion requires the appropriate chain
rule; do not merely relabel the time axis.

### 3.3 W1, W2, W2-squared and population matching

For uniform empirical measures, let `Pi` contain nonnegative transport plans
whose row/column masses are `1/N` and `1/M`:

```text
W1 = min_{P in Pi} sum_ij P_ij ||x_i-y_j||_2
W2 = sqrt(min_{P in Pi} sum_ij P_ij ||x_i-y_j||_2^2)
W2_squared = min_{P in Pi} sum_ij P_ij ||x_i-y_j||_2^2
```

Flow2PC's `emd` means **W1**, full feature dimension, Euclidean cost, not sliced
Wasserstein. `ot.emd2` is the optimizer function name; it does not make every
call a W2-squared score. Cost construction and the final square root determine
the metric. W1 allows different population sizes. The native transport W2 helper
first truncates both arrays to their first `min(N,M)` rows; the shared transport
path reproduces that equal-size policy after its configured subset selection.

Transport permits optimal rematching of particles. It assesses population
agreement rather than preserving source-particle identities. A smaller student
W1/W2 does not prove a lower velocity-distillation error.

Report an unweighted mean across the specified evaluation times, not a
cell-count-weighted mean unless explicitly named as an additional statistic.
In particular, `mean(W2_t^2)` and `mean(W2_t)` can select different checkpoints;
`sqrt(mean(W2_t^2))` is not the usual reported mean W2 either.

Implementation: [distance and inverse-coordinate functions](../src/flow2pc/evaluation/ode.py),
[shared benchmark evaluator](../src/flow2pc/evaluation/metric_flow.py),
[dataset policies](../src/flow2pc/utils/experiment_protocols.py).

### 3.4 Three distinct selection questions

| Stage | What is being selected | Current meaning |
|---|---|---|
| Teacher training | An epoch/update within one teacher run | Method-specific: final ALI/shared-data OTP weights, native OTP selection callback, or MFM/MMFM validation-selected weights |
| Teacher sweep | A configuration among teacher runs | Must name the dataset's metric and complete sampling/time protocol; transport headline reporting is W2 |
| Student training/sweep | A velocity-distilled student | Saved velocity-validation score; W1/W2 are diagnostics, not the current student selection objective |

There is no universally meaningful teacher training-loss curve across GAN,
potential, geopath and spline methods. Keep their losses and validation methods
named separately. The literature's teacher benchmark does not define our PC
distillation checkpoint rule.

## 4. Single-cell CITE and Multi

### 4.1 Inputs, counts and representation

Use the processed H5AD files from the supplied release:

| Profile family | Relative path under `/data/lthung/ali-cfm-data` | State/time keys | Counts at days 2,3,4,7 |
|---|---|---|---|
| CITE | `single_cell/op_cite_inputs_0.h5ad` | `obsm['X_pca']`, `obs['day']` | `[7476,6999,9511,7254]` |
| Multi | `single_cell/op_train_multi_targets_0.h5ad` | Same | `[7264,8544,10933,6960]` |

The source is the [processed data release](https://data.mendeley.com/datasets/hhny5ff7yj/1).
Use the first D stored PCA components for D=5/50/100. Do not refit PCA, replace it
with raw gene counts, or call a 5D experiment gene-level imputation.

### 4.2 Training populations and folds

Run the two interpolation folds independently:

| Fold | Withheld index/day | Observed indices/days | Benchmark source |
|---|---|---|---|
| 1 | Index 1 / day 3 | `[0,2,3]` / days `[2,4,7]` | All day-2 cells, tau 0 to 1/3 |
| 2 | Index 2 / day 4 | `[0,1,3]` / days `[2,3,7]` | All day-3 cells, tau 1/3 to 2/3 |

Today the standard Flow2PC presets/retained teachers generally use **fold 1 and
seed 42**, not the full two-fold, multi-seed experiment. The local ALI driver
loops over both folds and its configured `seed_list`; its reproduction audit
uses seeds 42–46. Average folds within each seed before summarizing independent
training seeds. One fold or MC resampling seeds are not a five-training-run result.

Native ALI training loads **all cells at observed times**. It excludes the
withheld time from training objectives and observed-data metric references.
The shared OTP adaptation defaults to `training.population_split: all`, also
using full observed pools. These behaviors are distinct from the row partitions
recorded by the shared loader for student training and diagnostics.

### 4.3 Input preprocessing versus metric coordinates

The native path uses `preprocessing_protocol: native_all`:

1. Slice the stored PCA to the chosen D.
2. At 5D, apply `StandardScaler` over all four marginal populations, including
   the temporal holdout. Use population variance (`ddof=0`). At 50D/100D,
   the current native presets do not whiten.
3. Apply per-coordinate min/max normalization using all populations, preserving
   the native sequential float32 operation order.
4. Save both transforms. At scoring, undo min/max; **retain 5D whitening** for
   the benchmark W1, but restore original PCA units at 50D/100D.

This is all-time preprocessing parity, not train-only preprocessing. Withheld
cells affect scaling statistics even though they do not enter teacher gradient
updates. Do not claim an untouched-data inductive pipeline. For a new train-only
study, every method needs a separately declared protocol and newly compatible
teacher artifacts; never silently rescale an existing teacher.

MFM/MMFM prepared profiles currently standardize on observed training rows.
Their `metric_space: dataset` W1 uses a separate all-population benchmark scaler
at 5D and raw PCA coordinates at higher D. Matching reporting units does not
make their training transforms/populations identical to native ALI/OTP.

### 4.4 Evaluation and images

Integrate the frozen velocity from the **preceding observed marginal's global
time** to the withheld global time, and compare its output population against
the complete withheld population using W1. The upstream single-cell evaluator
uses all source and reference cells and adaptive Dopri5, batching source cells
for memory. EMD uses uniform masses even when source/reference sizes differ.

Flow2PC's default pretraining YAMLs cap source and metric samples at 2,048. Those
are finite-budget diagnostics, not full-population table replicas. Record both
counts and seeds; use an identical evaluation budget across compared methods.

Save per-fold W1 and held-out reference/prediction plots, with raw-PCA scatter
views clearly labeled as projections. Full-D distances must use all D features.
Keep generator/interpolant predictions separate from ODE velocity predictions.

The generic pretraining evaluator currently integrates from time 0. That happens
to agree with fold 1, but **not fold 2**. The shared population evaluator contains
the preceding-observed branch. A later-fold result is not benchmark-equivalent
until its actual `source_index` and integration interval are verified.

Sources: [shared H5AD loader](../src/flow2pc/data/real/single_cell.py),
[preprocessing](../src/flow2pc/data/preprocessing.py),
`../ali-cfm/ali_cfm/scripts/train_ali_single_cell.py`,
`../ali-cfm/docs/SINGLE_CELL_IMPLEMENTATION_PLAN.md`.

## 5. Spatial transcriptomics: ST

### 5.1 Data and coordinates

Use `spatial_transcriptomics/aligned/aligned_spots/U*_tumor_coordinates.csv`
under the ALI data root. These are headerless aligned tumour-only coordinates
in **(y,x)** order, not gene-expression feature vectors.

| Index | Slide | Model time | Count |
|---|---|---:|---:|
| 0 | U2 | 0 | 755 |
| 1 | U3 | 1/3 | 870 |
| 2 | U4 | 2/3 | 945 |
| 3 | U5 | 1 | 1034 |

Use the existing U5 reference frame. The supplied handoff records image
downscaling by ten, landmark-based registration, and tumour purity strictly
greater than 0.8. Do not substitute unaligned coordinates, all-spot tables or
different purity thresholds. Background images are visualization assets, not
input covariates. Slide ordering does not establish elapsed biological time.

### 5.2 Training and evaluation

Leave out U3 or U4 separately. Current configs retain U3. Native training uses
all rows at the remaining slides with all-slide min/max normalization to [0,1]
and no whitening. Row splits in saved metadata do not reduce those native
teacher training populations.

The recovered ALI training branch uses the original piecewise-reference
regularizer and matching ST loops; it does not replace them with the newer
`picewise_oskar_version` branch. The current recipe has 1,000 regression
pretraining updates, 70,000 GAN updates and 10,000 CFM updates, with ALI batch
256 and CFM batch 128. Detailed rates are in section 9.

The source ST evaluator pushes the preceding slide to the withheld slide using
Dopri5 and a 101-point requested time grid; scores are W1 after restoring aligned
coordinate units. A requested grid is not the adaptive solver's internal step
count. Current generic Flow2PC pretraining instead configures RK4/100 substeps,
all source particles, a 512-point metric cap, and earliest-origin rollout.
These differences must be disclosed or harmonized before a paper-table claim.

For U4, the preceding source is U3 at tau 1/3, not U2 at zero. Preserve that
interval across methods. Save the withheld-slide observation/prediction overlay
on the matching clean aligned H&E image, converting (y,x) to display (x,y).
Default backgrounds are in `spatial_transcriptomics/aligned/ref_U5_warped_images`.

Sources: [ST loader](../src/flow2pc/data/temporal.py),
[ST pretrain config](../configs/pretrain/alicfm/alicfm_st.yaml),
[recovered ST method notes](ALICFM_DATASETS.md),
`../ali-cfm/ali_cfm/scripts/train_ali_st.py`.

## 6. Knot: preserve the supplied code variant

### 6.1 Exact configured distribution

**The `+1` lift is intentional and present in your ALI-CFM code.** Both
`../ali-cfm/train_teacher.py:get_knot_data` and the data bundle's
`knot/knot_distribution.py:y` use it. Flow2PC's implementation preserves it.
It is not a migration error and must not be removed when MFM/MMFM adopt this data.

Current shared settings: `K=120`, `samples_per_time=10`, `noise_std=0.01`,
raw 2D coordinates, no whitening/min/max scaling, no withheld times. For
`r=linspace(0,3,K)` and `u=r-1.5`, split u into three consecutive equal blocks:

```text
block 1: mean_x = 3*(u+0.5);        mean_y = -0.5*tanh(5*(u+1))+0.5
block 2: mean_x = cos(2*pi*(u-.75)); mean_y = sin(2*pi*(u-.75))+1
block 3: mean_x = 3*(u-0.5);        mean_y =  0.5*tanh(5*(u-1))+0.5
tau = r/3
X_tau ~ Normal(mean(tau), noise_std^2 * I_2)
```

K must be divisible by three. Noise is independent across coordinates and
observations; row positions do not establish ground-truth trajectories.
The bundled `loop_distribution` uses NumPy's global RNG and returns endpoint
arrays `(10,2)`, interiors `(10,K-2,2)`, and a K-entry model-time array.
Flow2PC stores marginals as K separate `(samples_per_time,2)` arrays using its
seeded snapshot generator; the same marginal distribution does not imply
bitwise identity with the bundled generator's draw order.

### 6.2 Fresh teacher batches versus the finite snapshot export

The restored native ALI teacher creates **fresh full noisy curves per batch**:
128 independently drawn curves, each spanning all 120 knots, then float32
conversion. The configured ten stored points per time are not a limit of ten
unique observations across the entire teacher run. NumPy x-noise/y-noise draw
order and native update order are preserved in the migrated knot trainer.

The shared OTP knot adaptation also creates fresh independent analytic noise
at all knots. By contrast, current MFM/MMFM prepared knot profiles export a
fixed ten-point snapshot per time, split approximately 90/10 into training and
validation. That is **not an identical observation budget or generator regime**.
Preserve the same generated observations or the same agreed fresh-sampling
protocol before calling a comparison controlled.

### 6.3 Evaluation and the paper/code distinction

Our local variant follows the supplied checkout and its training defaults.
It differs from the paper's printed benchmark in K, noise and the lifted
middle branch. Record all three; do not label this variant an exact published
score reproduction or silently change it to the printed formula.

Current saved snapshots have ten rows per time; the native shared split has
8 train/1 val/1 test. Native ALI knot training does not train against only
those eight stored rows. A one-row reserved snapshot is inadequate as the sole
calibration reference. No time is withheld in the ordinary knot config: its W1
is reconstruction/interpolation diagnostics over observed times, not a new
temporal-generalization score.

For a stronger analytic diagnostic, fresh independent observations can be drawn
from the same formula with separately declared evaluation seeds and budgets.
Specify whether the ODE starts from the finite saved source or fresh analytic
initial noise. Those initial laws differ and must not be silently interchanged.
The E3 analytic-knot experiment is a separately documented protocol.

Save a time-versus-x/y trajectory view, the 2D curve with time coloring, and
observed/model clouds at declared knots. Include the original generator
settings, source count and noise scale in captions and metrics.

Sources: [shared knot generator](../src/flow2pc/data/synthetic/knot.py),
[native fresh-batch sampler](../src/flow2pc/pretrain/models/alicfm/_source/knot_training.py),
[knot config](../configs/pretrain/alicfm/alicfm_knot.yaml),
`/data/lthung/ali-cfm-data/knot/{README.md,knot_distribution.py}`.

## 7. Gulf of Mexico and Beijing: OTP-origin protocols

### 7.1 GOM inputs and splits

Use `/data/lthung/otp-fm/gom/marginals.npz` (safe, no pickle), or the equivalent
source `GoMvortex_data.npy`. The source array is float64 `(9,111,2)`; the portable
NPZ is float32, matching the native loader cast. It contains 999 observed states.
These are separate unpaired observations; row i at two times is not a track.

| Role | Indices | Model times |
|---|---|---|
| Training, including endpoints | `[0,2,4,6,8]` | `[0,.25,.5,.75,1]` |
| Held-out interpolation | `[1,3,5,7]` | `[.125,.375,.625,.875]` |
| Intermediate observed marginals/potential locations | `[2,4,6]` | `[.25,.5,.75]` |
| Reported training-distance times, source excluded | `[2,4,6,8]` | `[.25,.5,.75,1]` |

Fit a per-coordinate StandardScaler to all 999 source observations, including
holdouts, with `ddof=0`; train only at the observed indices. No additional row
validation split is in the native transport experiment. Saved arrays remain
raw. Inverse-transform both predictions and references before W2. The bundle
does not establish longitude/latitude units or observed trajectory identities.

The portable provenance pins the processed source to
`YunyiShen/SB-Iterative-Reference-Refinement`, commit
`5d91ad36c65e4f46ff16202d0c788619a7538bda`,
`Notebooks/data/GoMvortex_data.npy`. Do not regenerate a different ocean sample
or download the full oceanographic archive to replace this benchmark.

### 7.2 Beijing inputs and splits

Use `/data/lthung/otp-fm/beijing/marginals.npz`: flat float32 `values` shape
`(9409,1)`, offsets shape `(14,)`, and the saved time/split arrays. It represents
hourly Dingling PM2.5 measurements grouped by selected calendar month. Grouping
does **not** mean replacing the readings by one monthly mean.

| Index | Month | Count | Role |
|---|---|---:|---|
| 0 | 2013-03 | 744 | Train/source |
| 1 | 2013-05 | 706 | Train |
| 2 | 2013-07 | 710 | Held out |
| 3 | 2013-09 | 716 | Train |
| 4 | 2013-11 | 720 | Train |
| 5 | 2014-01 | 721 | Held out |
| 6 | 2014-03 | 743 | Train |
| 7 | 2014-05 | 734 | Train |
| 8 | 2014-07 | 728 | Held out |
| 9 | 2014-09 | 702 | Train |
| 10 | 2014-11 | 717 | Train |
| 11 | 2015-01 | 730 | Held out |
| 12 | 2015-03 | 738 | Train/target |

The source is the UCI Beijing Multi-Site Air Quality archive, using only
`PRSA_Data_Dingling_20130301-20170228.csv` and its PM2.5 column. Drop missing
PM2.5, sort chronologically, and take every other month from March 2013 through
March 2015. Training indices are `[0,1,3,4,6,7,9,10,12]`, holdouts `[2,5,8,11]`.
Model time is `i/12`; physical metadata is the calendar-month index `2*i`.

Fit one StandardScaler over all 9,409 readings, including holdouts. Native
training has no extra observation-level validation split. Report W2 in original
micrograms/m^3 after inverse scaling, and average training distances over
`[1,3,4,6,7,9,10,12]`, excluding source index 0.

Beijing is **1D** for every method using this benchmark. Other station columns,
other pollutants, meteorology, log1p transforms or clipping negative predictions
define different experiments. Existing checkpoints cannot be relabeled as
six-variable models. Same-time missing-coordinate E3.2 is undefined for this
state; time alone is not another measured state variable.

### 7.3 W2 reporting, native inference and time matching

Report W2 separately at each of the four temporal holdouts, then their
unweighted arithmetic mean. Also report the noninitial observed-time mean.
Keep the held-out and training means distinct.

| Evaluation path | Dynamics | Which time is scored? | Meaning |
|---|---|---|---|
| Native OTP GOM | EMA finite-interval consistency sampler | Minimum W2 among stored grid indices `nearest-3` through `nearest+3`, clipped to the grid, at holdouts; nearest only at observed times | Exact parity with the supplied native implementation |
| Native OTP Beijing | Same native sampler | Nearest stored grid point for each target time; no nearby-time minimum | Supplied Beijing implementation |
| Shared ODE evaluator | Integrate instantaneous diagonal EMA velocity `v_ema(x,t,t)` | Exact requested observation time | Common ODE comparison; not the same sampler/time rule as native OTP |

**Decision for this handoff: follow the paper-defined held-out times for the
main cross-method comparison.** Evaluate predictions at the specified model
times, without optimizing a nearby prediction time against the reference.
Record solver/interpolation tolerances if a method has a discrete sampling
grid. The paper text specifies the target times; it does not prescribe the
source helper's nearby-time minimum. Exact-time scoring is our operational
interpretation of that target-time benchmark, not a claim that the paper
specifies every numerical solver setting.

The native GOM minimum uses the reference distribution to choose a nearby
prediction time and is more permissive. Preserve it only as a separately named
author-code parity diagnostic. Native sampler versus instantaneous ODE is an
additional distinction: if using common ODE inference, label it explicitly and
do not claim it reproduces the native few-step experiment just because W2 and
time indices match. A native finite-step method can also be evaluated at the
specified times, but the current nearby-search score is not that report.

Default native YAMLs use 25 GOM inference steps/111 source particles and 20
Beijing steps/up to 1,000 particles (the source has only 744 rows). These are
the current recipe settings, not a universal solver budget. Shared ALI transport
configs use all source rows, metric cap 1,024, Dopri5 and tolerance 1e-5.

### 7.4 Teacher checkpoint selection

The source reproduction callback defaults to `mean_w2`; it additionally supports
`mean_squared_w2`, which is the paper-described squared-distance selection.
**The paper-described teacher checkpoint rule for the main transport experiment
is `mean_squared_w2`; the reported table distance is still mean W2.** Default
transport YAMLs currently omit that key and therefore use `mean_w2`, a documented
implementation mismatch rather than the prescribed paper selection.
When squared selection is requested, `best_mean_w2.pt` is retained separately
from `best.pt`. Read `selection_rule` and `selection_score` in the run, not just
the filename. Reporting W2 remains correct even when selection uses W2-squared.

ALI transport's corrected sweep ranks **mean raw-coordinate exact-time W2**
(`teacher_heldout_w2`, protocol `transport_ode_w2_v1`), with W1 retained only as
a diagnostic. Older documents/exports may still contain the previous W1-only
scores. W1 values must never be renamed to W2, and default mean-W2 selection
must not be described as the squared-selection rule.

That ALI sweep ranking is not the same selection rule as mean W2-squared either.
The current MFM/MMFM method-specific validation selectors must likewise be
disclosed if retained. Before claiming a new fully aligned transport teacher
comparison, make those choices explicit; this document does not rewrite their
trained artifacts or silently change their YAMLs.

The temporal holdouts are used for checkpoint/configuration selection in these
paths. They are validation/tuning populations for those selected models, not an
untouched final test. They remain excluded from training gradient updates.

### 7.5 Method-specific training and visualizations

OTP's `W2` recipe name refers to precomputed OT pairing chains, while `W2Inf`
uses independent pairings; both can use a `W2InfPotential` implementation. The
potential name, coupling and reported W2 metric are separate concepts. Preserve
the saved strength/width/kernel/curriculum/EMA settings for each method.

ALI on GOM/Beijing is a dataset adaptation of its original staged objectives,
not a published ALI result on these datasets. MFM/MMFM also need explicitly
bound adaptation configs. All must use the specified input marginals and the
agreed evaluation, without replacing their original training objectives.

GOM: save raw-coordinate observation/prediction clouds, trajectory views and
per-time W2, visibly marking holdouts. Beijing: save concentration distributions
and time-marginal comparisons with concentration units, without silently
clamping outputs. Native figures use nearest stored times, which can differ
from the nearby-time prediction selected for GOM's scalar score.

Sources: [transport loader](../src/flow2pc/data/real/transport.py),
[native OTP profile/time rules](../src/flow2pc/pretrain/models/otpfm/profiles.py),
[native result metadata](../src/flow2pc/pretrain/models/otpfm/evaluation.py),
[selection callback](../src/flow2pc/pretrain/models/otpfm/_source/reproduction.py),
[transport correction](TRANSPORT_W2_CORRECTION.md), and the two supplied data READMEs.

## 8. Cell tracking

### 8.1 Source and observation budget

Use sequence `01`, frames `t000.tif`–`t114.tif`, and the corresponding
`01_ST/SEG` segmentation masks from PhC-C2DH-U373. Select **label 4**. Inputs
are spatial support pixels of that segmented cell, not tracked pixel identities.
The prepared convenience export
`cell_tracking/prepared/mask_cell4_stack.npy` is binary `(115,520,696)`.
Convert each mask with `argwhere(mask)[:, [1,0]]` to model **(x,y)** order.

The ALI/OTP native-profile comparison uses a **fixed ten-point sample per frame**,
chosen without replacement before model initialization and recorded in the run.
Use the exact saved subset indices across methods to isolate model differences.
Do not replace ten points with all pixels and call it the same training budget.
All 115 times are observed; there is no withheld temporal frame in this recipe.

Native scaling is all-frame min/max to [0,1], with no whitening. A teacher may
explicitly predict in raw coordinates (`cfm_coordinates: raw`); its adapter must
convert velocities consistently before shared-model integration. Historical
ALI tracking uses a separately bound successful checkpoint and metadata, not
an assumed match to the current retraining YAML.

### 8.2 Full-mask ODE evaluation and overlays

Start from **all** pixels of the initial label-4 mask and integrate to:

| Displayed frame (1-based) | Stored index | Model time | Full reference pixels in current source |
|---:|---:|---:|---:|
| 1 | 0 | 0 | 4154 |
| 25 | 24 | 24/114 | 4873 |
| 50 | 49 | 49/114 | 5831 |
| 75 | 74 | 74/114 | 4917 |
| 100 | 99 | 99/114 | 4209 |
| 115 | 114 | 1 | 6001 |

These source counts are metadata from this bundle; do not assume they are the
ten-point training population or that the predicted pixel count grows to match
the reference. ODE transport preserves the number of source particles. Uniform
W1 can compare those unequal counts.

Convert predictions back to pixels. Overlay them on the original clean image
with a downward image y-axis, observed mask support, shared axes, and identical
source particles for teacher/student. Keep the six-panel overlay, centroid/time
trajectory view and per-frame W1 diagnostics. The initial frame is a source
sanity check; average predictive distances over the **five noninitial selected
frames**, not over all 115 frames and not including the trivial initial score.
The paper's qualitative figure is not itself a published numerical W1 table.

**Current alignment gaps:** the generic ALI evaluator still has an eight-frame
fallback `[0,16,32,48,65,81,97,114]`. Some other paths score every time while their
figures show six. Explicitly set `[0,24,49,74,99,114]` for both metrics and plots.
The current MFM/MMFM prepared profiles use full mask pools with a 90/10 row
split, not the native ten-point observation budget. Their available artifacts
therefore need a separate comparison label until training subsets are aligned.

Randomly reserving pixels from the same frame/cell does not create independent
biological examples. Describe within-mask reconstruction/spatial generalization
and avoid treating those pixels as independent subjects for confidence intervals.

Sources: [mask loader](../src/flow2pc/data/temporal.py),
[figure renderer/default frames](../src/flow2pc/evaluation/figures.py),
[ALI tracking recipe](../configs/pretrain/alicfm/alicfm_cell_tracking.yaml),
[OTP tracking recipe](../configs/pretrain/otpfm/otpfm_cell_tracking.yaml),
`/data/lthung/ali-cfm-data/cell_tracking/README.md`.

## 9. Current native-family training recipes: implementation snapshot

These values are from **current YAML/source**, not claims that every final
published hyperparameter was specified or every retained checkpoint used them.
All have explicit seeds; ALI entries below use seed 42. Staged ALI counts are
configured loop budgets; an upstream loop's inclusive end can affect the exact
update count, so save the actual completed steps too.

| ALI profile | Widths CFM/G/D | Pretrain / GAN / CFM budgets | Batch CFM / ALI | LR pretrain / G / D / CFM | CFM coupling; regularizer |
|---|---|---|---|---|---|
| CITE/Multi 5D | 64/64/64 | 2000 / 70000 / 2001 | 128 / 128 | .001 / .00005 / .00005 / .001 | `mmot`; LAND second derivative |
| CITE/Multi 50D/100D | 1024/1024/1024 | Same | Same | Same | Same |
| ST | 128/64/64 | 1000 / 70000 / 10000 | 128 / 256 | .001 / .00001 / .00001 / .001 | `full`; piecewise reference, coefficient 10 |
| Lifted knot | 32/128/128 | 0 / 50000 / 30000 | 128 / 128 | unused / .001 / .001 / .0001 | Independent endpoints; linear reference, coefficient 1 |
| Tracking retraining recipe | 256/256/256 | 0 / 70000 / 50000 | 128 / 128 | unused / .0001 / .0001 / .001 | `border`; linear reference, coefficient 1; ten points/frame |
| GOM adapted winner recipe | 256/256/256 | 2000 / 30000 / 10000 | 128 / 128 | .001 / .00005 / .00005 / .0001 | `border`; second derivative, coefficient .001 |
| Beijing adapted winner recipe | 256/256/256 | 2000 / 30000 / 10000 | 128 / 128 | .001 / .0005 / .0005 / .001 | `border`; zero regularizer coefficient |

CITE/Multi use R3GAN; ST/knot/tracking/GOM use the configured vanilla GAN;
Beijing uses R3GAN. Single-cell second-derivative coefficients, in order
CITE5/CITE50/CITE100/Multi5/Multi50/Multi100, are
`[5e-7,5e-8,5e-8,1e-7,5e-8,1e-8]`. They are local calibrated/released choices,
not recovered unpublished final coefficients. Single-cell time smoothing is
.01; ST/tracking .001; knot zero. Save coupling for every ALI stage, not only CFM.

| Native OTP recipe | Width/depth | Epochs | Batch | LR | Inference | Default within-run selector |
|---|---|---:|---:|---:|---|---|
| GOM W2 | 128 / 10 | 2000 | 64 | .001 | 25 steps, <=111 source rows | Mean held-out W2 unless explicitly overridden |
| Beijing W2 | 128 / 4 | 600 | 128 | .003 | 20 steps, <=744 actual source rows | Same |

These native OTP configs retain method-specific potential strengths/widths,
sigmoid curriculum, consistency objective and EMA. GOM's current W2 config
uses `meanflow`, Beijing `imf`; both native loss settings are `mse`, EMA .99.
MMD/KL recipes are separate; do not infer their schedules from the W2 table.

OTP on ALI-origin datasets is a **shared-data adaptation** selected by its own
sweeps. Its default uses original direct W2Inf solutions with independent tuples,
the configured MeanFlow/IMF objective, and final EMA export. It is not an
OTP-paper reproduction of ALI's experiments. `training.steps` is converted to
whole native epochs; `actual_updates` in `sampling_contract.json` can exceed
the requested count by the final partial epoch's rounding.

MFM/MMFM should use their own original architecture, losses and optimization
settings under the agreed dataset contract. Their present general presets are
adaptations, not automatic replicas of every baseline in the ALI/OTP papers.
For example, current MMFM defaults include adaptive Gaussian path noise;
this is not the same as declaring a noiseless cubic baseline.

Links: [ALI configs](../configs/pretrain/alicfm/),
[OTP configs](../configs/pretrain/otpfm/),
[MFM configs](../configs/pretrain/mfm/),
[MMFM configs](../configs/pretrain/mmfm/).

## 10. Validation and student distillation

The shared H5AD/snapshot loaders record approximately 90/5/5 train/val/test rows
at observed CITE/Multi/ST times. Tracking defaults use 85/10/5; knot's current
ten-row export uses 8/1/1. Whole withheld-time populations are placed in `test`.
The word `test` is a loader label, not proof that a population was untouched by
teacher selection or preprocessing.

**Native full-pool teachers can have seen observed rows later reserved for
student validation.** Student exclusion from its own gradient batches does not
make those rows unseen by its teacher. Fixed ten-point tracking subsets can also
overlap those later row labels. Keep those scopes explicit.

Current student objective is only

`mean((student.velocity(x,tau) - frozen_teacher.velocity(x,tau))^2)`.

The query law follows the frozen teacher's method/checkpoint: ALI generator and
training-compatible coupling, OTP's saved conditional construction, MFM geopath,
or MMFM interpolant. Sharing a dataset does not make all samplers identical.
Do not add direct-PC likelihood, endpoint penalties or source anchoring.

| Dataset | Default student validation scope | Main selection population |
|---|---|---|
| CITE/Multi/ST | `observed_rows` | Reserved observed-time rows under the chosen teacher-query construction |
| GOM/Beijing | `heldout_times` | Up to `validation_batches*validation_batch_size` cells/time; equal time weighting |
| Knot | `observed_rows` | Finite snapshot row reserve; very small with ten rows/time |
| Tracking | Observed-row checks plus existing tracking-specific checks when configured | Historical ALI selection can use the maximum teacher-energy-normalized global/late/final velocity error |

For the default dataset policy see [validation scope](../src/flow2pc/utils/experiment_protocols.py)
and [validation implementation](../src/flow2pc/distill/validation.py).
Tracking's normalized selection score is not interchangeable with raw MSE on
another dataset. MFM/MMFM validation loops can construct different query sets
from ALI/OTP even when their scope labels agree; record sampler, batches, seed
and validation time grid for fair within-cohort selection.

Use `best.pt` for the selected student's final diagnostics, and `last.pt` only
as the latest resumable state. Log `teacher_ode_*`, `student_ode_*` and
`direct_pc_*` separately. Direct-PC results are diagnostics, not the velocity
training objective or the density of the empirical-source ODE distribution.

## 11. MFM/MMFM compatibility gaps to resolve before equal-protocol claims

| Item | Native ALI/OTP paths | Prepared MFM/MMFM paths today | Required handling |
|---|---|---|---|
| Observed rows | Often full observed populations | Independent 90/10 train/validation split | Match pools/budgets for a controlled comparison, or report the difference |
| Input scaler | Native all-marginal scaler, sometimes whitening then min/max | Observed-training-row standardization | Same raw observations do not imply identical preprocessing; bind/report explicitly |
| Single-cell folds | Local source loops support folds 1 and 2; current YAMLs often fold 1 only | Catalog fixes holdout `[1]` | Do not present one-fold results as full leave-one-out aggregate |
| Tracking observations | Fixed ten points/frame in native profiles | Full-mask prepared populations | Align exact subsets for paper-budget runs |
| Knot training data | Fresh analytic samples in native ALI/shared OTP | Fixed prepared snapshots | Align generator regime, K, noise and lift |
| Transport distance | Correct primary is W2; historical W1 files remain | Shared evaluator can report both, native selection remains method-specific | Inspect the actual selected field, not a generic `emd` label |
| Later-fold inference | Paper/source uses preceding population | Common evaluator supports it; generic ALI pretraining runner starts at zero | Audit actual source/time interval |
| Tracking evaluation | Six requested panels | Some defaults use eight frames or all 115 | Set identical explicit scored/rendered indices |
| Checkpoint selection | Final-stage or native transport selector | Method-specific validation-selected weights | State within-run and across-run rules independently |

Code references: [prepared catalog](../src/flow2pc/data/mfm_catalog.py),
[prepared splitting/scaling](../src/flow2pc/data/mfm_snapshots.py),
[MFM snapshot training](../src/flow2pc/pretrain/models/mfm/snapshots.py),
[MMFM training](../src/flow2pc/pretrain/models/mmfm/trainer.py).

This section identifies differences; it does not authorize silent model changes
or retrofit new metadata onto existing weights. A future harmonized training
run needs an agreed contract before launching, while old artifacts preserve
their original bindings.

## 12. EB representation caveat

The bundle also contains `single_cell/ebdata_v3.h5ad`. Current shared ALI/OTP
`eb` uses **2D PHATE** (`obsm/X_phate`), with labels `Day 00-03`, `Day 06-09`,
`Day 12-15`, `Day 18-21`, `Day 24-27`; counts
`[2656,4374,3650,3873,3650]` and ordinal model times `[0,.25,.5,.75,1]`.
Current default holdout is index 1. Do not interpret an ordinal category index
as an exact sampling day or invent day-range midpoints.

The paper's EB PCA benchmark and the absent `eb_velocity_v5.npz` are different
inputs. MFM/MMFM's `eb_phate2`, `eb_pca5` and `eb_pca50` are distinct profiles;
only matching representations can share a comparison row. Current ALI EB
hyperparameters are an adaptation, not a verified EB paper reproduction.

## 13. Outputs, reproducibility and team handoff

Every method must retain enough information to reconstruct its experiment:

```text
dataset/profile + dimension + representation key
raw source paths/hashes + prepared-array identity
physical labels/units + model time grid + velocity-time convention
observed and withheld time indices + exact training subset/row indices
training/validation/test split policy + all RNG seeds
input transforms + fit populations + inverse/reporting transform
method loss/coupling/interpolant/potential settings + actual updates
teacher checkpoint path/hash + optional generator/geopath dependencies
within-run selector + across-run selector + score formula and units
ODE/native inference mode + solver/tolerance/steps/NFE limit
particle/reference counts + sampling seed + matching/truncation policy
evaluation source index + exact or searched scoring times
per-time results + aggregate + training-seed mean/sample standard deviation
visualizations + W&B/local tracking status
```

Standard stages remain `python main.py --config <yaml>`; dataset roots and
`output.dir` are configuration fields. Existing ALI/OTP defaults point to
`/data/lthung/ali-cfm-data` or `/data/lthung/otp-fm`; a moved checkpoint must
restore its saved preprocessing and use a compatible explicit data root.
MFM/MMFM prepared defaults live under `data/mfm/prepared/PROFILE/seedSEED`.
Do not copy raw observations or checkpoints into Git.

Use W&B project **flow2pc**, names containing method/dataset/dimension/fold/seed,
and distinct run output directories. Current ALI/OTP configs use `logging`,
while MFM/MMFM can use `wandb`; those are parser conventions, not different
projects. Local metrics and images remain available when remote logging fails.
Frozen reference collections in `/data/lthung/flow2pc/pretrained-teachers/` are
not evidence that every source run used today's YAML defaults.

Store the six-frame tracking list explicitly in new configs. For
single-cell/ST record whether the run is fold 1 or fold 2, not merely `heldout`.
For transport record `mean_w2` versus `mean_squared_w2` and native versus exact
time. For knot include `lift=+1`, K and sigma in the run description.

### Supplementary uncertainty experiments

E3.1/E3.2 are Flow2PC experiments, not the source papers' training/selection
protocols. See [experiment documentation](experiments/README.md) and
[E3.2 conditional protocol](experiments/E3_2_CONDITIONAL.md). Their HDR-ECE and
Energy Score do not replace benchmark W1/W2. Conditional KDE describes an
explicitly smoothed evaluation law; it is not automatically an exact flow or
PC likelihood. Real-data results can be retrospective because of teacher
selection/scaling reuse. Keep numerical-estimator sensitivity separate from
model uncertainty and independent training-seed variation.

### Verification performed for this document

Read-only inspection of versioned papers, supplied data READMEs, live source,
current YAMLs and saved manifest metadata. No model files, data, training configs
or numerical code were changed. No training, distillation, evaluation, data
preparation or tests were run. Source-code parity and metadata inspection do not
establish numerical reproduction of a published result.
