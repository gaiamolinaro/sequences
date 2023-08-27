# Based on Anne's randomize.m file
import numpy as np
import matlab.engine

# use matlab to call createstimsequence
use_matlab = True
if use_matlab:
    m = matlab.engine.start_matlab()
    m.addpath("C:/Gaia/CCN Lab/pseudoR Project/pseudoR Codes/pseudoRWMCtrl/", nargout=0)
else:
    from numpy.matlib import repmat

def shuffle_along_axis(arr, axis):
    idx = np.random.rand(*arr.shape).argsort(axis=axis)
    return np.take_along_axis(arr,idx,axis=axis)

num_conditions = 10 # number of different file sequences to generate (change if needed)
reps = 12 # number of stimulus repetitions after first presentation (change if needed)
to_dir = "C:/Gaia/CCN Lab/pseudoR Project/pseudoR Codes/pseudoRWMCtrl/sequences/"  # where to store the sequences

all_goal_sets = [[5, 17], [25, 26], [24, 30], [56, 59], [76, 77]]*2 # fractals to use in the learning phase, includes goal and non-goal

for si in range(num_conditions):
    blocks = np.array([6, 6, 6, 6, 6, 6, 6]) # 6 blocks: 3*type
    trial_types = []
    for tp in range(3):
        for tt in np.random.permutation(2):
            trial_types.append(tt)
    trial_types = np.array(trial_types)
    block_idx = np.hstack(((np.repeat(np.arange(1, len(blocks), 2), 2) + trial_types)))
    blocks = blocks[block_idx]

    max_ns = np.max(blocks)

    goal_set = all_goal_sets[si]
    # counterbalance which image is the goal and which one is the non-goal across participants
    goal_image = goal_set[int(si+1 >= num_conditions//2)]
    nongoal_image = goal_set[1-int(si+1 >= num_conditions//2)]

    # how many stimuli for which action (3 total) is correct + set size, organized by block
    # make sure that sometimes one action is totally incorrect (for ns 3)

    R = np.vstack((
        np.array([1, 2, 3, 6]),
        np.array([2, 2, 2, 6]), 
        np.array([1, 2, 3, 6]),
        np.array([2, 2, 2, 6]), 
        np.array([1, 2, 3, 6]),
        np.array([2, 2, 2, 6])
        ))

    R = np.column_stack((shuffle_along_axis(R[:, 0:3], axis=1), R[:, 3])) # mix up stim/action within the rule
    
    # order by permutated blocks
    realR = np.zeros_like(R)
    for ns in np.unique(blocks):
        T =  np.where(blocks==ns)
        realR[T] = R[np.where(R[:, 3]==ns)]
    R = realR

    # rules
    rules = {}
    for b in range(len(blocks)):
        rules[b] = [a for a in range(3) for i in range(R[b, a])]

    # stimulus sets (folders from where images will be taken for each block)
    stim_sets = np.random.permutation(len(blocks))+1
    
    # stimuli
    stimuli = {}
    for b in range(len(blocks)):
        stimuli[b] = (np.random.permutation(max_ns)+1)[0:blocks[b]]

    # stimulus sequences
    # create a prototype for each set size
    seqprototype = {}
    for ns in np.unique(blocks): # replaced with max ns variable
        if use_matlab:
            seqprototype[ns] = np.squeeze(np.array(m.createstimsequence(m.double(int(reps)), m.double(int(ns))))).astype(int)
        else:
            # worse alternative if createstimsequence doesn't work
            temp_seqprototype = repmat(np.arange(1, ns+1), 1, reps+1)
            np.random.shuffle(temp_seqprototype)
            seqprototype[ns] = temp_seqprototype

    # randomize which stimuli happen at which position
    stim_seqs = {}
    for b in range(len(blocks)):
        arr = np.random.permutation(blocks[b])+1
        stim_seqs[b] = np.squeeze(np.array([arr[i-1] for i in seqprototype[blocks[b]]]))

    # create csv
    # rows: stim, correct key, set size, blocks, img_folders, img_nums, trial_type, goal_img, nongoal_img
    goal_stim_count = 0
    for b in range(len(blocks)):
        set_size = blocks[b]
        block_length = (reps + 1) * set_size  # number of trials in a block
        this_block = np.full((9, block_length), np.nan)

        block_stims = stim_seqs[b]
        _, unique_idx = np.unique(block_stims, return_index=True)
        block_rules = rules[b]
        block_imgs = stimuli[b]
        block_cond = trial_types[b]
        
        this_block[0] = block_stims
        this_block[2] = np.repeat(set_size, block_length)  # set size
        this_block[3] = np.repeat(b+1, block_length)  # block number
        this_block[4] = np.repeat(stim_sets[b], block_length)  # image folder
        this_block[6] = np.repeat(block_cond, block_length)

        for st in range(set_size):
            this_block[1, np.where(block_stims==st+1)] = block_rules[st]  # correct key
            this_block[5, np.where(block_stims==st+1)] = block_imgs[st]  # stimulus image number
        
        if block_cond == 0:
            this_block[7] = goal_image
            this_block[8] = nongoal_image
            goal_stim_count += block_length

        if b == 0:
            train_seq = this_block
            unique_stims = this_block[:, unique_idx]
        else:
            train_seq = np.column_stack((train_seq, this_block))
            unique_stims = np.column_stack((unique_stims, this_block[:, unique_idx]))
        
    # For the last version of task, test sequence consisted of 3 subsequences which each contained all task stimuli 
    # in randomized order: testing presentation not counterbalanced like training seq
    test_seq = np.column_stack((unique_stims[:, np.random.permutation(unique_stims.shape[1])], 
                                unique_stims[:, np.random.permutation(unique_stims.shape[1])], 
                                unique_stims[:, np.random.permutation(unique_stims.shape[1])]))

    # checks
    # check that unique stims and metadata in test and train are the same
    short_train_seq = train_seq[0:6]
    short_test_seq = test_seq[0:6]
    unique_train, train_freq = np.unique(short_train_seq[:, short_train_seq[0].argsort()], return_counts=True, axis=1)
    unique_test, test_freq = np.unique(short_test_seq[:, short_test_seq[0].argsort()], return_counts=True, axis=1)
    same_stims = np.all(unique_train == unique_test)

    # check that each unique stim is repeated 13 times in train and 3 in test
    train_13 = np.all(np.isin(np.unique(train_freq), np.array([reps, reps+1, reps+2])))
    test_3 = np.all(np.unique(test_freq) == 3)

    # check that sequences are the correct length
    train_size_13 = train_seq.shape[1]//np.sum(blocks) == reps+1
    test_size_3 = test_seq.shape[1]//np.sum(blocks) == 3

    if same_stims and train_13 and test_3 and train_size_13 and test_size_3:
        np.savetxt(f"{to_dir}seq{si+1}_learning.csv", train_seq, delimiter=",")
        np.savetxt(f"{to_dir}seq{si+1}_testing.csv", test_seq, delimiter=",")
    else:
        if not same_stims:
            print(f"same_stims check failed for condition {si+1}")
        if not train_13:
            print(f"train_13 check failed for condition {si+1}")
        if not test_3:
            print(f"test_3 check failed for condition {si+1}")
        if not train_size_13:
            print(f"train_size_13 check failed for condition {si+1}")
        if not test_size_3:
            print(f"test_size_3 check failed for condition {si+1}")

# quit matlab
if use_matlab:
    m.exit()
