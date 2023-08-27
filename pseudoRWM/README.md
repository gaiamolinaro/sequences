# Sequences for pseudoRWM experiments

Across documents, each row is a variable and each column is a trial.

|Row|Variable|Description|
|---|---|---|
|0|stim|Stimulus number|
|1|correct_key|Index of the correct key to press|
|2|set size|Number of different stimuli in the block|
|3|blocks|Block number|
|4|img_folders|Folder from which stimuli are taken|
|5|img_nums|Number of the stimulus image used|
|6|trial_type|Type of trial (1 = Points, 0 = Goals)|
|7|goal_img|Number of the fractal image used (or NaN)|
|8|group|Only in some experiments, tells which stimulus group a stimulus is in|

## pseudoRWM
Standard version, new goals on every iteration.

## pseudoRWMCtrl
A single pair of goal/nongoal images (counterbalanced across participants).

## pseudoRWMReps
Manipulating the number of repetitions.
|Group|Description|
|---|---|
|0|Mostly repeated goals (75% repeated, 25% novel)|
|1|Mostly novel goals (25% repeated, 75% novel)|
|2|Half and half (50% repeated, 50% unique)|

## pseudoRWMConf
Conflicting goal/nongoal outcomes.
|Group|Description|
|---|---|
|0|Pure goals: outcomes are drawn from a set of two fractal image pairs, each used six times per stimulus with consistent labels|
|1|Conflicting goals: outcomes are presented presented three times as a goal and three times as a nongoal|
|2|Baseline: |