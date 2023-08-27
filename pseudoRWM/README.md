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

## pseudoRWM
Standard version, new goals on every iteration.

## pseudoRWMCtrl
A single pair of goal/nongoal images (counterbalanced across participants).

## pseudoRWMReps
Manipulating the number of repetitions.

## pseudoRWMConf
Conflicting goal/nongoal outcomes.