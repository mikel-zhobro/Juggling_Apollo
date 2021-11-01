import datetime
import json
import os

saveSimResults =True
resultsFolder = "../../"

statelogs = {
        'tau':[], 'q':[], 'dq':[], 't':[],
    }

data_raw = {
        'statelogs': statelogs
    }

filename = os.path.join(
        resultsFolder,
        '{}_simdata_raw.dat'.format(datetime.now().strftime("%y-%m-%d_%H-%M-%S"))
    )
if saveSimResults:

    with open(filename, 'w') as f:
        json.dump(data_raw, f)

    print('\nSAVED: {} data points saved to {}\n'.format(len(statelogs['q']), filename))
# TODO: maybe pickle