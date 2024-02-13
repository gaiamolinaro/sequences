# Sequences for pseudoRWM experiments

Across documents, each row is a variable and each column is a trial.

|Row|Variable|Description|
|---|---|---|
|0|stim|Stimulus number*|
|1|correct_key|Index of the correct key to press|
|2|set_size|Number of different stimuli in the block|
|3|block|Block number|
|4|img_folder|Folder from which stimuli are taken|
|5|img_num|Number of the stimulus image used\*\*|
|6|trial_type|Type of trial (1 = Points, 0 = Goals)|
|7|goal_img|Number of the fractal image used (or NaN) for goals|
|8|nongoal_img|Number of the fractal image used (or NaN) for nongoals|
|9|group|Only in some experiments, tells which stimulus group a stimulus is in|

\*Used as the image number pseudoRWM and pseudoRWMCtrl! <br>
\*\*Never actually used in pseudoRWM and pseudoRWMCtrl!

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
|0|Pure goals: outcomes are drawn from a set of 2 fractal image pairs, each used 6 times per stimulus with consistent labels|
|1|Conflicting goals: outcomes are presented presented 3 times as a goal and 3 times as a nongoal|
|2|Baseline: outcomes are drawn from a set of 4 fractal image pairs, each used 6 times per stimulus with consistent labels|

Note: "Novel" fractals were in fact repeated within goal conditions in pseudoRWMReps and pseudoRWMConf.

## pseudoRWMReps2
Manipulating the number of repetitions more strongly.
|Group|Description|
|---|---|
|0|Mostly repeated goals: 9 presentations of the shared goal/nongoal pair, 3 novel (per stimulus)|
|1|Mostly novel goals:  3 presentations of the shared goal/nongoal pair, 6 novel (per stimulus)|
|2|Half and half: 6 presentations of the shared goal/nongoal pair, 6 novel (per stimulus)|

Shared goal/nongoal images are stable within blocks across stimulus groups, and new on each block. 

## pseudoRWMConf2
Conflicting goal/nongoal outcomes more strongly.
|Group|Description|
|---|---|
|0|Pure goals: outcomes are drawn from a single pair of fractal image used 12 times per stimulus with consistent labels|
|1|Conflicting goals:  outcomes are drawn from a single pair of fractal image used 12 times per stimulus, half the time with goal-nongoal labels and half the time with nongoal-goal labels labels|
|2|Baseline: outcomes are drawn from a set of 2 fractal image pairs, each used 6 times per stimulus with consistent labels|

"Novel" fractals are now truly novel (only occur once in the task) in both pseudoRWMReps2 and pseudoRWMCtrl2.