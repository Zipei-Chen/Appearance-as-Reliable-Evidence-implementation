import os
import pickle as pkl
import numpy as np

if __name__ == '__main__':

    male_path = '../body_models/smplx/SMPLX_MALE.pkl'
    female_path = '../body_models/smplx/SMPLX_FEMALE.pkl'
    neutral_path = '../body_models/smplx/SMPLX_NEUTRAL.pkl'

    data_m = pkl.load(open(male_path, 'rb'), encoding='latin1')
    data_f = pkl.load(open(female_path, 'rb'), encoding='latin1')
    data_n = pkl.load(open(neutral_path, 'rb'), encoding='latin1')

    if not os.path.exists('../body_models/misc_smplx'):
        os.makedirs('../body_models/misc_smplx')

    np.savez('../body_models/misc_smplx/faces.npz', faces=data_m['f'].astype(np.int64))
    np.savez('../body_models/misc_smplx/J_regressors.npz', male=data_m['J_regressor'], female=data_f['J_regressor'], neutral=data_n['J_regressor'])
    np.savez('../body_models/misc_smplx/posedirs_all.npz', male=data_m['posedirs'], female=data_f['posedirs'], neutral=data_n['posedirs'])
    np.savez('../body_models/misc_smplx/shapedirs_all.npz', male=data_m['shapedirs'], female=data_f['shapedirs'], neutral=data_n['shapedirs'])
    np.savez('../body_models/misc_smplx/skinning_weights_all.npz', male=data_m['weights'], female=data_f['weights'], neutral=data_n['weights'])
    np.savez('../body_models/misc_smplx/v_templates.npz', male=data_m['v_template'], female=data_f['v_template'], neutral=data_n['v_template'])
    np.save('../body_models/misc_smplx/kintree_table.npy', data_m['kintree_table'].astype(np.int32))
