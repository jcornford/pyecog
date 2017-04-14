
from sklearn.preprocessing import normalize
import numpy as np

class HmmVanilla():
    ''' Hmm class expecting transition matrix and an emission matrix'''
    def __init__(self,A, phi):
        '''
        :param A: transition matrix, A_ij is p(Z_t=j|Z_t-1=i)
        :param phi: emission matrix, phi_ij is p(X_t=j|Z_t=i)
        '''

        assert all(np.sum(A, axis = 1))==1
        # transpose to get left eigen vector
        eigen_vals,eigen_vecs = np.linalg.eig(A.T)
        ind = np.where(eigen_vals ==1)[0]

        self.stationary_dist = eigen_vecs[:,ind].T[0] # transpose back should be row vec
        self.A = A
        self.phi = phi

    @staticmethod
    def forward(x,k,N,A,phi,stationary_dist):
        alpha  = np.zeros((k,N)) # init alpha vect to store alpha vals for each z_k (rows)
        alpha[:,0] = np.log(phi[:,y[0]] * stationary_dist)
        for t in np.arange(1,N):
            max_alpha_t = max(alpha[:,t-1]) # b
            exp_alpha_t = np.exp(alpha[:,t-1]-max_alpha_t) # exp sum over alphas - b
            alpha_t     = np.log(phi[:,y[t]]*exp_alpha_t.dot(A)) # write this dot out...
            alpha[:,t]  = alpha_t + max_alpha_t
        return alpha

    @staticmethod
    def backward(x,k,N,A,phi,stationary_dist,alpha):
        beta  = np.zeros((k,N))
        posterior  = np.zeros((k,N))
        beta[:,N-1] = 1 #minus one for pythons shit
        posterior[:,N-1] = np.exp(alpha[:,N-1]+beta[:,N-1])
        posterior[:,N-1] /= sum(posterior[:,N-1])

        for t in range(0,N-1)[::-1]:
            #print(t,end=',')
            max_beta_t   = max(beta[:,t+1])
            exp_beta_t   = np.exp(beta[:,t+1]-max_beta_t)
            beta_t       = np.log((phi[:,y[t+1]]*exp_beta_t).dot(A)) # is this correct?
            # phi inside the dot product as dependnds on the
            beta[:,t]    = beta_t
            posterior[:,t] = np.exp(alpha[:,t]+beta[:,t])
            posterior[:,t] /=sum(posterior[:,t])  # normalise as just proportional too...
        return beta, posterior

    def forward_backward(self,x):
        '''
        x is a vector of observations
        returns posterior distribution of p(Zt
        '''
        k = self.A.shape[0]
        N = x.shape[0]
        alpha = self.forward(x,k,N,self.A,self.phi,self.stationary_dist)
        beta,self.posterior = self.backward(x,k,N,self.A,self.phi,self.stationary_dist,alpha)
        return self.posterior

def get_state_emission_probs(emissions, annotated_states):
    n_states = len(np.unique(annotated_states))
    n_emiss  = len(np.unique(emissions))
    emis_mat = np.zeros(shape= (n_states,n_emiss))

    for i,label in enumerate(annotated_states.astype('int')):
        emis = emissions.astype('int')[i]
        emis_mat[label, emis] += 1
        emis_probs = normalize(emis_mat, axis = 1, norm='l1')
    return emis_probs

def get_state_transition_probs(labels):
    if len (labels.shape) > 1:
            labels = np.ravel(labels)

    tp = np.zeros(shape= (2,2)) # todo why is this is hardcoded?
    for i, label in enumerate(labels[:-1]):
        next_label = int(labels[i+1])
        label = int(label)
        tp[label,next_label] += 1

    tp = normalize(tp, axis = 1, norm='l1')
    return tp
