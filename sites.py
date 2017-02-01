# Create a Markov model for site data.
import numpy as np

transitions = {}"""first we create a dictionary to store 
all our transitions this will store, as a key, the start
page and the end page, and then its going to count how many
times that happened and then after that we're going to turn
it into a probability distribution. to do that, we're going 
to need another dictionary called row_sums. """
row_sums = {}"""this dictionary is just going to have the 
start state as the key and then we can devide all the transitions
that start with that start state by this row_sums."""

# collect counts
for line in open('site_data.csv'):# we loop through every line in site_data
    s, e = line.rstrip().split(',')# we get the start and end page
    transitions[(s, e)] = transitions.get((s, e), 0."""the default value is 0""") + 1# then we populate our transition dictionary --> s,e is the key
    row_sums[s] = row_sums.get(s, 0.) + 1

# normalize all the counts so that they become real probability distribution
for k, v in transitions.iteritems():
    s, e = k # split out the key
    transitions[k] = v / row_sums[s] # split the value to whatever it currently is divided by the row sums

# calculate the initial state distribution
print "initial state distribution:"
for k, v in transitions.iteritems():# loop through all the transitions
    s, e = k #split out the key
    if s == '-1': # if this is the start of a sequence
        print e, v # print out end page and value

# which page has the highest bounce? -> we want to find everything in the dictionary where the end state is B
for k, v in transitions.iteritems():
    s, e = k
    if e == 'B':
        print "bounce rate for %s: %s" % (s, v) # print out bounce rate for this state
