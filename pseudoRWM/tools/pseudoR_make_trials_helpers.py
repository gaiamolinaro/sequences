import numpy as np
import matplotlib.pyplot as plt
# create sequence with controlled delays
# original from Anne Collins
def create_stim_sequence(reps, num_stim):
    reps = 8
    num_stim = 6
    seq = np.tile(np.arange(num_stim), (1, reps))
    beta = 5
    criterion = False
    ps = []
    while not criterion:
        delays = 1 + np.floor(np.ones((1, 2 * num_stim - 1)) * reps/2)
        criterion = True
        count = reps * np.ones(num_stim)+1
        print(count)
        seq = np.arange(num_stim)
        last = np.arange(num_stim)
        for t in range(num_stim*(reps+1)):
            Q = np.zeros(num_stim)
            L = np.zeros(num_stim)
            for i in range(num_stim):
                idx = int((t-last[i])+count[i])
                #print(idx)
                #print(delays[0][idx])
                #Q[i] = delays[0][t-last[i]+count[0][i]]
                #L[i] = t - last[i]


create_stim_sequence(2, 4)

'''

while ~criterion
    for t=ns+1:ns * (reps + 1)
    Q = [];
    for i=1:ns
    Q(i) = delays(t - last(i)) + count(i);
    L(i) = t - last(i);
end
if max(L) == size(delays, 2)
    [~, choice] = max(L);
else
    softmax = exp(beta * Q);
    softmax = softmax. / sum(softmax);
ps = [0 cumsum(softmax)];
choice = find(ps < rand);
choice = choice(end);
end
seq(t) = choice;
last(choice) = t;
delays(L(choice)) = delays(L(choice)) - 1;
count(choice) = count(choice) - 1;
end

delay = [];
last = zeros(1, ns);
alldelays = [];
dseq = [];
for t=1:length(seq)
if last(seq(t)) > 0
    alldelays = [alldelays t - last(seq(t))];
    dseq = [dseq seq(t)];
end
last(seq(t)) = t;
end
ds = unique(alldelays);
distr = [];
for d=ds
distr(d) = sum(alldelays == d);
end
% distr = distr / sum(distr);
[h, p, st] = chi2gof(alldelays, 'edges', [1:length(distr) + 1]-.5, ...
'expected', mean(distr) * ones(1, length(distr)));

figure(1)
plot(distr, 'o-')
criterion = p > .05;

criterion = (max(distr) - min(distr)) < 2;
for d=ds
for i=1:ns
distr(i, d) = sum(alldelays == d & dseq == i);
end
end
hold
on
distr = sum(distr(:,:));
plot(distr
','
o - ')
hold
off

criterion = criterion & ((max(distr) - min(distr)) < 2);
end
end
'''