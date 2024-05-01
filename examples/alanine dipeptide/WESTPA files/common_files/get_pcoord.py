import numpy as np
import os
import torch

filename = "COLVAR"

# Open the file
with open(filename, 'r') as infile:
    lines = infile.readlines()

    pos = 0
    data = []
    for line in lines:
        sline = line.split(' ')  # separates line into a list of items.  ',' tells it to split the lines at the commas
        if pos == 0:
            data.append(sline[2:])
        elif sline[0] != '#!':
            try:
                data.append([float(value) for value in sline[1:]])
            except ValueError:
                pass
        pos += 1

colvar_data = np.array(data[1:])[:, 1:]

# get the input data for SPIB
traj_data = colvar_data[:,:-4]

np.save('traj_data.npy', traj_data)

WE_folder = os.environ.get('WEST_SIM_ROOT')

model_path = WE_folder + '/SPIB/SPIB.model'

# check whether there is a saved SPIB_model
if os.path.isfile(model_path):
    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(model_path)

    model.to(device)

    label, _, z_latent, _ = model.transform(traj_data, batch_size=4096, to_numpy=True)
else:
    # if there is no save SPIB model, use the default values as z_latent
    label = np.zeros(traj_data.shape[0]).astype(int)
    z_latent = np.zeros([traj_data.shape[0], 2])

# get pcoord in angstrom
pcoord = np.concatenate([colvar_data[:, -4:-2], z_latent], axis=-1)
np.savetxt("pcoord.dat", pcoord)
np.savetxt("label.dat", label, fmt='%i')