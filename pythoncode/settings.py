import numpy as np

dy = '/Users/jonathan/PhD/Seizure_related/2015_08_PyRanalysis/classifier_writeup/'

f129 = np.loadtxt(dy + 'features_orginal301_n129.csv', delimiter=',')
l129 = np.loadtxt(dy + 'labels_orginal301_n129.csv', delimiter=',')

np.savetxt('../features_orginal301_n129.csv',f129,delimiter=',')
np.savetxt('../labels_orginal301_n129.csv',l129,delimiter=',')

f342 = np.loadtxt(dy + 'features0616_n342.csv', delimiter=',')
l342 = np.loadtxt(dy + 'labels0616_n342.csv', delimiter=',')

np.savetxt('../features0616_n342.csv',f342,delimiter=',')
np.savetxt('../labels0616_n342.csv',l342,delimiter=',')

## Validation set ##
val_feats = np.loadtxt(dy + 'test_features_sept_n279.csv', delimiter=',')
val_labels = np.loadtxt(dy + 'sept_279_labels.csv', delimiter=',')
val_uncertain = np.loadtxt(dy + 'sept_279_uncertain_events.csv',delimiter=',')

val_uncertain = np.array([int(x) for x in val_uncertain])
ok_indexes = []
for i in range(val_feats.shape[0]):
    if i not in val_uncertain:
        ok_indexes.append(i)

val_feats_fair = val_feats[ok_indexes,:]
val_labels_fair = val_labels[ok_indexes]

np.savetxt('../val_feats_fair.csv',val_feats_fair,delimiter=',')
np.savetxt('../val_labels_fair.csv',val_labels_fair ,delimiter=',')
