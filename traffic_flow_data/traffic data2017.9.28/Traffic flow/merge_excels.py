import os
import pandas as pd
import numpy as np
import sys


excel_src = sys.argv[1]

excels = [i for i in os.listdir(excel_Src) if i.split('.')[-1] == 'xlsx']
for exl in excels:
    data.append(pd.read_excel(exl))

flows = []
    for i in data:
        flows.append(np.hstack((np.array(i['5 Minutes']).reshape(i.shape[0], 1), np.array(i['Flow (Veh/5 Minutes)']).reshape(i.shape[0], 1))))
flows = np.vstack(flows)

