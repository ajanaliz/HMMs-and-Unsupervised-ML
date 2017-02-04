# Discrete Hidden Markov Model (HMM)
import numpy as np
import matplotlib.pyplot as plt

"""the random_normalized function is going to create a valid 
random markov matrix by dimension d1 by d2"""
def random_normalized(d1, d2):
    x = np.random.random((d1, d2))
	#lets make sure all the rows sum to one
    return x / x.sum(axis=1, keepdims=True)


class HMM:
	#class constructor will take in the number of hidden states
    def __init__(self, M):
        self.M = M # number of hidden states
    """this function will take in X = the observables and it
	takes in a max_iter which control how many iterations of
	expectation maximization its going to do"""
    def fit(self, X, max_iter=30):
		"""include a seed here so that you can test your algorithm 
		and compare it to the other files. This can help you verify
		that your code is correct."""
        np.random.seed(123)
        # train the HMM model using the Baum-Welch algorithm
        # a specific instance of the expectation-maximization algorithm

        # determine V, the vocabulary size
        # assume observables are already integers from 0..V-1
        # just like when we're doing supervised learning, when we expect the classes to be labeled 0 to K-1
		# X is a jagged array of observed sequences
		V = max(max(x) for x in X) + 1
        N = len(X) # store number of sequences in N

        self.pi = np.ones(self.M) / self.M # initial state distribution
        # A reasonable initialization for pi would be the uniform distribution
		self.A = random_normalized(self.M, self.M) # state transition matrix
        self.B = random_normalized(self.M, V) # output distribution

        print "initial A:", self.A
        print "initial B:", self.B

        costs = []
        for it in xrange(max_iter): # main loop
            if it % 10 == 0:
                print "it:", it # print every 10 epocs
            """we're going to keep an array for all our alphas and all our betas.
			we cant keep them in a matrix because they may be different lengths."""
			alphas = []
            betas = []
			"""although, we can keep all our probabilities in a numpy array of size N."""
            P = np.zeros(N)
            for n in xrange(N): # loop through each observation.
                x = X[n] # x will be the nth observation.
                T = len(x) # T will be the length of x
                alpha = np.zeros((T, self.M)) # init alpha
                alpha[0] = self.pi*self.B[:,x[0]] # pi times b for all states and the first observation.
                for t in xrange(1, T): # loop through each time after the initial time.
                    tmp1 = alpha[t-1].dot(self.A) * self.B[:, x[t]] # alpha at t becomes alpha at t-1 dot product with A and then done a element by element multiplication with B of x[t].
                    # tmp2 = np.zeros(self.M)
                    # for i in xrange(self.M):
                    #     for j in xrange(self.M):
                    #         tmp2[j] += alpha[t-1,i] * self.A[i,j] * self.B[j, x[t]]
                    # print "diff:", np.abs(tmp1 - tmp2).sum()
                    # assert(np.abs(tmp1 - tmp2).sum() < 10e-10)
                    alpha[t] = tmp1
                P[n] = alpha[-1].sum() # calculate the probability of the sequence, which is just the sum of the last alphas
                alphas.append(alpha) # append the alpha for current observation to our list of alphas
				
				# now do the same for beta.
                beta = np.zeros((T, self.M))
                beta[-1] = 1 # initial value for beta is 1.
                for t in xrange(T - 2, -1, -1): # loop through all the other times. And remember, for beta we count backwards.
                    beta[t] = self.A.dot(self.B[:, x[t+1]] * beta[t+1])
                betas.append(beta) # append the beta for current observation to our list of betas

            # print "P:", P
            # break
            assert(np.all(P > 0))
            cost = np.sum(np.log(P)) # calculate the total cost, which is the sum of the logs of P. --> thats the total log likelihood. 
            costs.append(cost) # append the cost for current observation to our list of costs
			
			"""Once you've calculated alpha and beta, you can re-estimate pi, A and B."""
            
            # now re-estimate pi, A, B
            self.pi = np.sum((alphas[n][0] * betas[n][0])/P[n] for n in xrange(N)) / N
            # print "self.pi:", self.pi
            # break
			
			# keep track of all my denominators and numerators for A and B updates.
			den1 = np.zeros((self.M, 1))
            den2 = np.zeros((self.M, 1))
            a_num = 0
            b_num = 0
            for n in xrange(N): # loop through all the samples.
                x = X[n] # x will be the nth observation.
                T = len(x) # T will be the length of x.
                # print "den shape:", den.shape
                # test = (alphas[n][:-1] * betas[n][:-1]).sum(axis=0, keepdims=True).T
                # print "shape (alphas[n][:-1] * betas[n][:-1]).sum(axis=0): ", test.shape
                den1 += (alphas[n][:-1] * betas[n][:-1]).sum(axis=0, keepdims=True).T / P[n]
                den2 += (alphas[n] * betas[n]).sum(axis=0, keepdims=True).T / P[n]

                # tmp2 = np.zeros((self.M, 1))
                # for i in xrange(self.M):
                #     for t in xrange(T-1):
                #         tmp2[i] += alphas[n][t,i] * betas[n][t,i]
                # tmp2 /= P[n]
                # # print "diff:", np.abs(tmp1 - tmp2).sum()
                # assert(np.abs(tmp1 - tmp2).sum() < 10e-10)
                # den += tmp1

                # update numerator for A
                a_num_n = np.zeros((self.M, self.M))
                for i in xrange(self.M):
                    for j in xrange(self.M):
                        for t in xrange(T-1): # all times except the last time.
                            a_num_n[i,j] += alphas[n][t,i] * self.A[i,j] * self.B[j, x[t+1]] * betas[n][t+1,j]
                a_num += a_num_n / P[n]

                # update numerator for B
                 b_num_n = np.zeros((self.M, V))
                 for i in xrange(self.M): # loop through every state.
                     for j in xrange(V): # loop through every possible observation.
                         for t in xrange(T): # loop through every time.
                             if x[t] == j: # if x[t] is equal to this observation.
                                 b_num_n[i,j] += alphas[n][t][i] * betas[n][t][i]
                
				
				# update the numerator for B.
				# b_num_n2 = np.zeros((self.M, V))
                # for i in xrange(self.M): # loop through every state.
                #     for t in xrange(T): # loop through every possible observation
                #         b_num_n2[i,x[t]] += alphas[n][t,i] * betas[n][t,i]
                # assert(np.abs(b_num_n - b_num_n2).sum() < 10e-10)
                b_num += b_num_n / P[n]
            # tmp1 = a_num / den1
            # tmp2 = np.zeros(a_num.shape)
            # for i in xrange(self.M):
            #     for j in xrange(self.M):
            #         tmp2[i,j] = a_num[i,j] / den1[i]
            # print "diff:", np.abs(tmp1 - tmp2).sum()
            # print "tmp1:", tmp1
            # print "tmp2:", tmp2
            # assert(np.abs(tmp1 - tmp2).sum() < 10e-10)
            
			# define the new A and B.
			self.A = a_num / den1
            self.B = b_num / den2
            # print "P:", P
            # break
        print "A:", self.A
        print "B:", self.B
        print "pi:", self.pi

		# plot the costs.
        plt.plot(costs)
        plt.show()

	
    def likelihood(self, x):
        # returns log P(x | model)
        # using the forward part of the forward-backward algorithm
        T = len(x) # T will be the length of the observation x.
        alpha = np.zeros((T, self.M))
        alpha[0] = self.pi*self.B[:,x[0]]
        for t in xrange(1, T): # loop through all the other times and update alpha, recursively.
            alpha[t] = alpha[t-1].dot(self.A) * self.B[:, x[t]]
        return alpha[-1].sum()

	# this will calculate all the likelihoods of every observation
    def likelihood_multi(self, X):
        return np.array([self.likelihood(x) for x in X])
	
	# this will return the log likelihood of the above function 
    def log_likelihood_multi(self, X):
        return np.log(self.likelihood_multi(X))

	# The Viterbi Algorithm:
    def get_state_sequence(self, x): # takes in one observable sequence
        # returns the most likely state sequence given observed sequence x
        # using the Viterbi algorithm
        T = len(x) # T will be the length of the observation x.
        # create delta and psi.
		delta = np.zeros((T, self.M)) 
        psi = np.zeros((T, self.M))
		# initial delta:
        delta[0] = self.pi*self.B[:,x[0]]
        for t in xrange(1, T): # loop through every other time.
            for j in xrange(self.M): # loop through all the states.
                #update delta and psi.
				delta[t,j] = np.max(delta[t-1]*self.A[:,j]) * self.B[j, x[t]]
                psi[t,j] = np.argmax(delta[t-1]*self.A[:,j])

        # backtrack
        states = np.zeros(T, dtype=np.int32) # sequence of length T. we want these to be indices, so we'll set them to ints.
        states[T-1] = np.argmax(delta[T-1])
        for t in xrange(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        return states

# this function will load our data and train an HMM.
def fit_coin():
    X = []
    for line in open('coin_data.txt'):
        # 1 for H, 0 for T --> this is because our HMM expects our symbols to be 0 to V-1.
        x = [1 if e == 'H' else 0 for e in line.rstrip()]
        X.append(x)

    hmm = HMM(2) # number of hidden states is 2(we know this since we generated the data).
    hmm.fit(X) # fit the observations
    L = hmm.log_likelihood_multi(X).sum() # return the likelihood(we need to sum it because they're all seperate).
    print "Log Likelihood with fitted params:", L

	"""lets do something interesting, lets set the HMM to the actual values"""
	
    # try true values
    hmm.pi = np.array([0.5, 0.5])
    hmm.A = np.array([[0.1, 0.9], [0.8, 0.2]])
    hmm.B = np.array([[0.6, 0.4], [0.3, 0.7]])
    L = hmm.log_likelihood_multi(X).sum()
    print "Log Likelihood with true params:", L

    # try viterbi
    print "Best state sequence for:", X[0]
    print hmm.get_state_sequence(X[0])


if __name__ == '__main__':
    fit_coin()
