#### GAIA MOLINARO, SEPT 13TH 2021 ####
#### Create task sequences for the pseudoR task ####
# See McDougle et al., 2021 for the original task
# Based on Sam's "create_task.m" code
# Only create sequences for fractal condition

# Load packages
import numpy as np
import pandas as pd
import matlab_subs
from itertools import groupby
from pseudoR_tf_to_csv import *

# Task parameters
bandit_set = 8  # number of bandits to use # Sam had 16
goal_set = 120  # number of fractals to use in the learning phase, includes goal and non-goal
prop_new_vs_old = 0.5  # proportion of new vs old images shown
tot_fract_set = goal_set*2  # total number of fractals available
# Blocks (each block has a learning and a testing phase)
num_blocks = 1
blocks = [i for i in range(0, num_blocks)]
pairs_per_block = 4  # how many pairs of bandits are simultaneously learned each block
reps = 30  # reps of each unique bandit pair # Sam had 30
learning_trials_per_block = pairs_per_block * reps  # total number of learning trials
num_subs = 100  # 4  # how many subjects will we run in this task? should be a multiple of 4
test_reps = 3  # number of repetitions for each bandit pair in the test phase
# Create vectors of for each participant
# Generate random sequences for half of the subjects
# Counterbalance steps: we want to have half of subs have exact opposite goal versus non-goal image sets
# We also want half of the people to have exact opposite old versus new image sets
# In Sam's version, first half of the subjects are counterbalanced relative to the other half
# Here, even/odd subjects are the counterbalanced version of each other
# To get Sam's version, initialize loop as for i in range(0, num_subs//2): instead of for i in range(0, num_subs, 2):
# and stim_vec[i+(num_subs//2)] = np.append(fh, sh) instead of stim_vec[i+1] = np.append(fh, sh)
stim_vec = np.empty((num_subs, goal_set))  # initialize

for i in range(0, num_subs, 4):  # floor division a//b = int(a/b)
    # first pair of subjects have same images for the task but they are opposite in value (goal vs no goal)
    shuffled_fract_set = np.random.permutation(tot_fract_set)
    learning_fract_set = shuffled_fract_set[:goal_set]
    stim_vec[i] = learning_fract_set  # first half of subs
    shifted1 = np.roll(stim_vec[i], goal_set // 2)  # new (shifted/flipped) vector
    fh1 = shifted1[0:len(shifted1)//2]  # first half
    np.random.shuffle(fh1)
    sh1 = shifted1[len(shifted1)//2:]  # second half
    np.random.shuffle(sh1)  # shuffle
    stim_vec[i+1] = np.append(fh1, sh1)  # counterbalanced subject group
    # second pair of subjects have opposite images for the task compared to the first pair
    learning_fract_set = shuffled_fract_set[goal_set:]
    stim_vec[i+2] = learning_fract_set  # first half of subs
    shifted2 = np.roll(stim_vec[i+2], goal_set // 2)  # new (shifted/flipped) vector
    fh2 = shifted2[0:len(shifted2) // 2]  # first half
    np.random.shuffle(fh2)
    sh2 = shifted2[len(shifted2) // 2:]  # second half
    np.random.shuffle(sh2)  # shuffle
    stim_vec[i+3] = np.append(fh2, sh2)  # counterbalanced subject group
    # note: if prop_new_vs_old < 1, not all available images will be used for the task
# Main loop
for si in range(num_subs):
    print(si)
    tf = {}
    ### MAIN TASK
    # bandit stim IDs
    shuffled_bandits = np.random.permutation(bandit_set)
    temp1 = [1 + sb for sb in shuffled_bandits[:bandit_set//2]]  # get first half of bandits, +1 because python is 0 indexed
    temp2 = [1 + sb for sb in shuffled_bandits[bandit_set//2:]]  # get second half of bandits, +1 because python is 0 indexed
    pairs = np.vstack((np.transpose(np.reshape(temp1, (2, bandit_set//4))),
                       (np.transpose(np.reshape(temp2, (2, bandit_set//4))))))  # line up pairs
    # everyone does shapes
    tf["bandit_stim"] = pairs
    # only if blocks = 2 or higher
    '''
    if si % 2:
        tf["bandit_stim"][0] = pairs[0:len(pairs)//2]
        tf["bandit_stim"][1] = pairs[len(pairs)//2:]
    else:
        tf["bandit_stim"][1] = pairs[0:len(pairs)//2]
        tf["bandit_stim"][0] = pairs[len(pairs)//2:]
    '''

    # specification of key for bandit ID matrix and associated probs / trial types
    tf["probs"] = [.9, .77]  # high prob events
    #tf["design_probs"] = [tf["probs"][0], 1 - tf["probs"][0],
    #                      tf["probs"][1], 1 - tf["probs"][1]]
    # added to the version of the task with both points and goals
    tf["design_probs"] = [tf["probs"][0], 1 - tf["probs"][0],
                          tf["probs"][1], 1 - tf["probs"][1],
                          tf["probs"][0], 1 - tf["probs"][0],
                          tf["probs"][1], 1 - tf["probs"][1]]
    #tf["design_type"] = [['G', 'G'], ['G', 'G']]
    # added to the version of the task with both points and goals
    tf["design_type"] = [['$', '$'], ['G', 'G'], ['$', '$'], ['G', 'G']]

    ## LEARNING PHASE
    # bandit sequence, trial_type, goal stimuli, which side better bandit is on
    '''
    for j in range(blocks):
        tf["seq"][j] = createstimsequence(reps - 1, pairs_per_block);
        while length(tf.seq(j, tf.seq(j,: ) == 1)) != reps
                or length(tf.seq(j, tf.seq(j,:) == 2)) != reps
                or length(tf.seq(j, tf.seq(j,:) == 3)) != reps
                or length(tf.seq(j, tf.seq(j,:) == 4)) != reps
                    tf.seq(j,:) = createstimsequence(reps - 1, pairs_per_block);
    '''
    # initialize
    tf["seq"] = [[] for j in range(num_blocks)]
    tf["stim_id"] = [[] for j in range(num_blocks)]
    tf["corr_action"] = np.zeros((len(blocks), learning_trials_per_block))
    tf["trial_type"] = np.zeros((len(blocks), learning_trials_per_block))
    tf["goal_stim"] = np.zeros((len(blocks), learning_trials_per_block))
    tf["nongoal_stim"] = np.zeros((len(blocks), learning_trials_per_block))
    tf["bandit1_seq"] = np.zeros((len(blocks), learning_trials_per_block))
    tf["bandit2_seq"] = np.zeros((len(blocks), learning_trials_per_block))

    for j in blocks:
        consec_occurOK = False
        while not consec_occurOK:
            tf["seq"][j] = []
            tf["stim_id"][j] = []
            for i in range(reps//2):
                miniseq = np.random.permutation(pairs_per_block*2)
                for mi in miniseq:
                    tf["stim_id"][j].append(mi)
            # transform into sequence:
            mi_to_seqi = {}
            count = 0
            for i in range(0, pairs_per_block*2, 2):
                mi_to_seqi[i] = count
                mi_to_seqi[i+1] = count
                count += 1
            for i in tf["stim_id"][j]:
                tf["seq"][j].append(mi_to_seqi[i])

            # check that there are no more than 3 occurrences of the same type (old or new) in a row
            groups = groupby(tf["seq"][j])
            consec_occur = [sum(1 for bandit in group) for label, group in groups]
            max_consec_occur = max(consec_occur)
            if max_consec_occur <= 2:
                consec_occurOK = True

        # correct bandit side of screen (0 for left, 1 for right)
        tempside = np.zeros_like(tf["seq"][j])  # initialize an array of all zeros
        # get random indices at which to place ones instead of zero
        tempside = [0 if i % 2 else 1 for i in tf["stim_id"][j]]  # odd is left, even is right side correct
        tf["corr_action"][j] = tempside  # store in corr_action

        tf["seq"] = np.array(tf["seq"])
        tf["stim_id"] = np.array(tf["stim_id"])

        count = 0  # counter for drawing from goal stimuli ID vector
        for t in range(learning_trials_per_block):
            # get goal trials stimuli
            #'''' # commented out if all trials are goal bandits
            if (tf["seq"][j][t]) % 2:  # odd number stim pairs are MONEY BANDITS
                tf["trial_type"][j, t] = 0
                tf["goal_stim"][j, t] = float("nan")
                tf["nongoal_stim"][j, t] = float("nan")
            else:  # even number stim pairs are GOAL BANDITS
            #'''
                tf["trial_type"][j, t] = 1
                tf["goal_stim"][j, t] = stim_vec[si, count]  # snag novel goal stim
                tf["nongoal_stim"][j, t] = stim_vec[si, count + goal_set//2]  # snag novel non - goal stim #(from second half of ID vector)
                count += 1  # iterate ID counter

            # specific bandit sequence
            #print(np.shape(tf["bandit1_seq"]))
            #print(np.shape(tf["bandit2_seq"]))
            #print(tf["bandit_stim"][tf["seq"][j][t], 0])
            #print(tf["seq"][j])
            #print(tf["bandit_stim"])
            tf["bandit1_seq"][j, t] = tf["bandit_stim"][tf["seq"][j][t], 0]
            tf["bandit2_seq"][j, t] = tf["bandit_stim"][tf["seq"][j][t], 1]
            '''
            # If there are multiple blocks
            tf["bandit1_seq"][j, t] = tf["bandit_stim"][j][tf["seq"][j][t], 0]
            tf["bandit2_seq"][j, t] = tf["bandit_stim"][j][tf["seq"][j][t], 1]
            '''

    # outcome sequences, pre - defined
    tf["highp_event"] = np.empty((blocks[-1]+1, learning_trials_per_block))  # does the high probability trial ge rewarded?
    tf["trial_probs"] = np.empty((blocks[-1]+1, learning_trials_per_block))  # probability that the good bandit gets rewarded

    for j in blocks:
        idx0 = [idx for idx, element in enumerate(tf["seq"][j]) if element == 0]  # pull out index of $$ stim 0
        idx1 = [idx for idx, element in enumerate(tf["seq"][j]) if element == 1]  # pull out index of $$ stim 1
        idx2 = [idx for idx, element in enumerate(tf["seq"][j]) if element == 2]  # pull out index of $$ stim 2
        idx3 = [idx for idx, element in enumerate(tf["seq"][j]) if element == 3]  # pull out index of $$ stim 3

    # while loop to make sure probabilities are exact
    # and that there are not too many repeats of the same bandit in a row
    probOK = False
    tempprb = np.zeros_like(idx0)

    while not probOK:  # matched reward sequences for the two 90% bandits

        for k in range(len(idx0)):
            x = np.random.random()  # get random float between 0 and 1
            if x < tf["probs"][0]:  # high prob
                tempprb[k] = 1
            else:
                tempprb[k] = 0
        for count, idx in enumerate(idx0):
            tf["highp_event"][j, idx] = tempprb[count]
            # added to the version of the task with both points and goals
            tf["highp_event"][j, idx1[count]] = tempprb[count]

            tf["trial_probs"][j, idx] = tf["probs"][0]
            # added to the version of the task with both points and goals
            tf["trial_probs"][j, idx1[count]] = tf["probs"][0]

        # matched reward sequences for the two 77% bandits
        for k in range(len(idx0)):
            x = np.random.random()  # get random float between 0 and 1
            if x < tf["probs"][1]:  # low prob
                tempprb[k] = 1
            else:
                tempprb[k] = 0

        for count, idx in enumerate(idx2):  # use idx2 if there are points too, use idx1 if it's just goals
            tf["highp_event"][j, idx] = tempprb[count]
            # added to the version of the task with both points and goals
            tf["highp_event"][j, idx3[count]] = tempprb[count]

            tf["trial_probs"][j, idx] = tf["probs"][1]
            # added to the version of the task with both points and goals
            tf["trial_probs"][j, idx3[count]] = tf["probs"][1]

        # use idx2 if there are points too, use idx1 if it's just goals
        if np.mean(tf["highp_event"][j, idx0]) == tf["probs"][0] \
            and round(np.mean(tf["highp_event"][j, idx2]), 2) == tf["probs"][1]:

            probOK = True
    ## ITIs and ISIs
    tf["isi_learning"] = [[] for j in range(num_blocks)]
    tf["iti_learning"] = [[] for j in range(num_blocks)]
    tf["stim_time"] = [[] for j in range(num_blocks)]
    for j in blocks:
        # learning
        # isi
        isi_bounds = [.5, 2.5]
        tf["isi_learning"][j] = (isi_bounds[0] + np.dot((isi_bounds[1] - isi_bounds[0]), np.random.rand(1, learning_trials_per_block)))[0]
        # iti
        iti_bounds = [1, 3]
        tf["iti_learning"][j] = (iti_bounds[0] + np.dot((iti_bounds[1] - iti_bounds[0]), np.random.rand(1, learning_trials_per_block)))[0]
        # stimulus time
        #times = rectpulse([1.2 1.4 1.6 1.8 2], learning_trials_per_block / 5);
        times_list = [1.2, 1.4, 1.6, 1.8, 2]
        times = []
        for time in times_list:
            for t in range(learning_trials_per_block//len(times_list)):
                times.append(time)
        np.random.shuffle(times)
        tf["stim_time"][j] = times

    tf["isi_learning"] = np.array(tf["isi_learning"])
    tf["iti_learning"] = np.array(tf["iti_learning"])
    tf["stim_time"] = np.array(tf["stim_time"])

    ## TESTING PHASE
    test_trials_per_block = test_reps*matlab_subs.nchoosek(np.size(tf["bandit_stim"]), 2)
    #test_trials_per_block = test_reps*matlab_subs.nchoosek(np.size(tf["bandit_stim"][j]), 2) # for more than 1 block

    tf["test_seq"] = [[] for j in range(num_blocks)]
    tf["test_bandit_order"] = np.zeros((len(blocks), test_trials_per_block))
    tf["iti_testing"] = np.zeros((len(blocks), test_trials_per_block))

    for j in blocks:
        # first define  vector  of each pairing
        bandits = []
        for arr in tf["bandit_stim"]:
        #for arr in tf["bandit_stim"] # for more than 1 block
            for y in arr:
                bandits.append(y)

        combos = matlab_subs.combnk(bandits, 2)
        randorder = np.random.permutation(test_trials_per_block)
        temptest = np.vstack((combos, combos))
        for i in range(test_reps - 2):  # number of lines like these depends on reps for test phase
            temptest = np.vstack((temptest, combos))  # test_reps trials all combos (without replacement = (8 * 7) / 2)
        coltemp1 = [b[0] for b in temptest]
        coltemp1_shuffled = [coltemp1[i] for i in randorder]
        coltemp2 = [b[1] for b in temptest]
        coltemp2_shuffled = [coltemp2[i] for i in randorder]
        tf["test_seq"][j] = np.array([coltemp1_shuffled, coltemp2_shuffled])

        # bandits on which side of  screen
        tempside = np.zeros(test_trials_per_block)  # initialize an array of all zeros
        idxside = np.random.permutation(np.size(tempside))  # get random indices at which to place ones instead of zero
        tempside[np.where(idxside <= (np.size(idxside) / 2))] = 1  # substitute based on idxside
        tf["test_bandit_order"][j] = tempside  # store in test_bandit_order

        # testing ITIs
        tf["iti_testing"][j] = iti_bounds[0] + np.dot((iti_bounds[1] - iti_bounds[0]), np.random.rand(1, len(coltemp1)))

    ### MEMORY TEST
    tf["mem_seq"] = [[] for j in range(num_blocks)]
    tf["corr_on"] = [[] for j in range(num_blocks)]
    tf["corr_gng"] = [[] for j in range(num_blocks)]

    for j in blocks:
        mem_seqOK = 0
        while not mem_seqOK:
            old_imgs = [stim for stim in stim_vec[si]]
            new_imgs = [i for i in range(tot_fract_set) if i not in old_imgs]
            new_imgs = np.random.choice(new_imgs, size=int(goal_set*prop_new_vs_old), replace=False)
            all_imgs = np.concatenate((old_imgs, new_imgs), axis=None)
            np.random.shuffle(all_imgs)
            tf["mem_seq"][j] = all_imgs
            tf["corr_on"][j] = [0 if stim in stim_vec[si] else 1 for stim in tf["mem_seq"][j]]
            # check that there are no more than 5 occurrences of the same type (old or new) in a row
            groups = groupby(tf["corr_on"][j])
            max_consec_occur = max([sum(1 for corr_act in group) for label, group in groups])
            if max_consec_occur <= 6:
                mem_seqOK = True
        # set the correct action for whether the goal vs nongoal question
        for stim in tf["mem_seq"][j]:
            if stim in tf["goal_stim"][j]:
                tf["corr_gng"][j].append(0)
            elif stim in tf["nongoal_stim"][j]:
                tf["corr_gng"][j].append(1)
            else:
                tf["corr_gng"][j].append(float("nan"))
            groups = groupby(tf["corr_gng"][j])
            max_consec_occur = max([sum(1 for corr_act in group) for label, group in groups])
    tf["corr_on"] = np.array(tf["corr_on"])
    tf["corr_gng"] = np.array(tf["corr_gng"])

    #'''
    # Produce a csv file
    dir = "C:/Gaia/CCN Lab/pseudoR Project/pseudoR Task Materials/new_sequences_with_points/"
    for j in blocks:
        tf_to_task_seq(tf, si, j, f"{dir}pseudoR/")
        tf_to_mem_seq(tf, si, j, f"{dir}pseudoR_memtest/")
    #'''