
from sklearn.preprocessing import normalize
import numpy as np

class HMM():
    ''' Hmm class expecting transition matrix only...
    '''
    def __init__(self,A):
        '''
        :param A: transition matrix, A_ij is p(Z_t=j|Z_t-1=i)
        '''

        assert all(np.sum(A, axis = 1))==1
        # transpose to get left eigen vector
        eigen_vals,eigen_vecs = np.linalg.eig(A.T)
        ind = np.where(eigen_vals ==1)[0]

        self.stationary_dist = eigen_vecs[:,ind].T[0] # transpose back should be row vec
        self.A = A

    @staticmethod
    def forward(x,k,N,A,phi,stationary_dist):
        alpha  = np.zeros((k,N))                           # init alpha vect to store alpha vals for each z_k (rows)
        alpha[:,0] = np.log((phi[:,0] * stationary_dist))
        for t in np.arange(1,N):
            max_alpha_t = max(alpha[:,t-1])                #  alphas are alredy logs, therefreo exp to cancel
            exp_alpha_t = np.exp(alpha[:,t-1]-max_alpha_t) # exp sum over alphas - b
            alpha_t     = phi[:,t]*(exp_alpha_t.T.dot(A))  # sure no undeflow here...
            alpha[:,t]  = np.log(alpha_t) + max_alpha_t    # take log and add back max (already in logs)
                                                           # this may be so small there is an overflow?
        return alpha

    @staticmethod
    def calc_phi(x,stationary_dist):
        phi = np.zeros(x.shape)
        for t in range(x.shape[1]):
            phi[:,t] = x[:,t]#/stationary_dist
        return phi

    @staticmethod
    def backward(x,k,N,A,phi,stationary_dist,alpha):
        beta  = np.zeros((k,N))
        posterior  = np.zeros((k,N))
        beta[:,N-1] = 1 # minus one for pythons indexing
        posterior_t = np.exp(alpha[:,N-1]+beta[:,N-1])
        posterior_t /= sum(posterior_t)
        posterior[:,N-1] = posterior_t

        for t in range(0,N-1)[::-1]: # python actually starts N-2 if [::-1]
            #print(t,end=',')
            max_beta_t   = max(beta[:,t+1]) # previous beta
            exp_beta_t   = np.exp(beta[:,t+1]-max_beta_t)
            beta_t       = A.dot((phi[:,t+1]*exp_beta_t))# is this correct?
            # phi inside the dot product as dependnds on the
            beta[:,t]    = np.log(beta_t)
            posterior_t  = np.exp(alpha[:,t]+beta[:,t])
            posterior_t /=sum(posterior_t)  # normalise as just proportional too...
            posterior[:,t] = posterior_t
        return beta, posterior

    @staticmethod
    def calc_phi_from_emission_matrix(x,phi_mat,stationary_dist):
        phi    = np.zeros((phi_mat.shape[0],x.shape[0]))
        for t in range(x.shape[0]):
            phi[:,t] = phi_mat[:,x[t]]
        return phi

    def forward_backward(self,x, phi_mat = None):
        '''
        If provide phi_mat, x is assumed to be a 1d vector of emissions. Else, if phi_mat = None,
        assumed x is a 2d vector of p(zt|xt)

        x is a vector of p(zt|xt)
        x_i is hidden state (rows)
        x_it t is the timepoint
        returns posterior distribution of p(Zt
        '''
        if phi_mat is None:
            self.phi =self.calc_phi(x,self.stationary_dist)
            k = x.shape[0]
            N = x.shape[1]
        else:
            self.phi =self.calc_phi_from_emission_matrix(x,phi_mat,self.stationary_dist)
            N = x.shape[0]
            k = phi_mat.shape[0]


        self.alpha = self.forward(x,k,N,self.A,self.phi,self.stationary_dist)
        self.beta, self.posterior = self.backward(x,k,N,self.A,self.phi,self.stationary_dist,self.alpha)
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
