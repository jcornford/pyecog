'''

Make this into a class!
CHANGE TO POMEMGRANATE - CHECK WORKS ON PIP
'''
import pomegranate
from sklearn.preprocessing import normalize
import numpy as np

def make_hmm_model(emission_mat, transition_probs):
    model = pomegranate.HiddenMarkovModel('ndf')

    ictal_emissions    = {i:emission_mat[1,i] for i in range(emission_mat.shape[1])}
    baseline_emissions = {i:emission_mat[0,i] for i in range(emission_mat.shape[1])}

    ictal    = pomegranate.State(pomegranate.DiscreteDistribution(ictal_emissions   ), name = '1')
    baseline = pomegranate.State(pomegranate.DiscreteDistribution(baseline_emissions), name = '0')

    model.add_state(ictal)
    model.add_state(baseline)

    model.add_transition( model.start, ictal, 0.05 )
    model.add_transition( model.start, baseline, 99.95)

    model.add_transition( baseline, baseline, transition_probs[0,0] )
    model.add_transition( baseline, ictal,    transition_probs[0,1]  )
    model.add_transition( ictal, ictal   ,    transition_probs[1,1] )
    model.add_transition( ictal, baseline,    transition_probs[1,0]  )

    model.bake(verbose=False )
    return model

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

    tp = np.zeros(shape= (2,2))
    for i, label in enumerate(labels[:-1]):
        next_label = int(labels[i+1])
        label = int(label)
        tp[label,next_label] += 1

    tp = normalize(tp, axis = 1, norm='l1')
    return tp
