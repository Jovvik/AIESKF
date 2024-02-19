import torch

torch.cuda.is_available = lambda: False

from pprint import pprint
from typing import Callable
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import random
import RnnModel
import Pipeline
import os
import scipy.io as scio
import functions
from datetime import datetime
# from Intergration import Deadreckoning, Error_state, Error_state_8, Error_state_8_LS
# %matplotlib widget

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7,8"  # Choose GPU core
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.manual_seed(3)  # make experiment repeatable
# =============================================================================
#                         Check GPU avaliability
# =============================================================================
if torch.cuda.is_available():
    dev = torch.device("cuda")  # cuda:1 cuda:2....etc.
    torch.set_default_dtype(torch.float64)
    datatype = np.float64
    torch.set_printoptions(precision=16)
    print("Running on the GPU")
else:
    dev = torch.device("cpu")
    torch.set_default_dtype(torch.float64)
    datatype = np.float64
    torch.set_printoptions(precision=16)
    print("Running on the CPU")


today_date = datetime.today().strftime("%m.%d.%y")[0:5].replace('.','')
currt_folder = os.getcwd()
# =============================================================================
#                      Step1 Load data and set parameter
# =============================================================================

#realdata_3d_10hz
#simudata_bias_3d_10hz
#simudata_MEMS_3d_10hz
#simudata_bias_White_3d_10hz
#GIVE_real_data
# GIVE_real_data_0316
#GIVE_real_data_0316_60s
# simudata_Errorfree_3d_10hz

data_paths = [currt_folder + '/data/' + 'Simu_MEMS_cut.csv',
              ]

indx_MB = True

# 
# operation = 'TRAIN'
  
operation = 'TEST'
in_model = '0212_MEMS_withgrad_shuffle_gnssgap_10lstm1105_200_pvab_noifimu0101noifatt00001_50Hz_Trajlength90_valoss' 
%matplotlib widget

# operation = 'TRAIN_USEOLD'
# in_model = '0212_MEMS_withgrad_shuffle_gnssgap_10lstm1105_200_pvab_noifimu0101noifatt00001_50Hz_Trajlength90_valoss'
# out_model = '0212_MEMS_withgrad_shuffle_gnssgap_10lstm1105_200_pvab_noifimu0101noifatt00001_50Hz_Trajlength90_valoss1'

# operation = 'MB_Only'

nograd_sd = True # True: with torch.no_grad, nograd    False: none, withgrad
idx_shuffle = True


# out_mat_file = "data/bias_pv_" + str(pos_add_error_gnss) +'_' + str(vel_add_error_gnss) + ".mat"

indx_train_gnssgap = True
gnssgap = 5 # every 5 seconds

idx_ifimu = False
idx_ifatt = False
idx_imu_scale = ['0.1','0.1']
idx_att_scale = '0.0001'

idx_feedback_type = 'pvab'
# idx_feedback_type = 'pvb'

idx_other_settings = 'newstruc'


# idx_other_settings = '_feedavpb_lkrelu_alltorchnorm_m2m_noifimuatt0101'
# idx_other_settings = '_m2m_loss123_' + str(pos_add_error_gnss) +'_' + str(vel_add_error_gnss)
idx_lossweight_coeff = ['1','10','5']

tarj_length = 90
target_Fs = 50


save_to_matlab = True
if operation == 'TEST':
    out_mat_file = "data/" + in_model + ".mat"


idx_train_batch_size = 20


fs_GT = 100
fs_GNSS = 1
fs_IMU = 100

GT_samp = int(fs_GT/target_Fs)
IMU_samp = int(fs_IMU/target_Fs)
Fs = target_Fs  # IMU freq
Fs_meas = fs_GNSS  # range meas freq



input_dim = 6 + 6 + 9 + 9
# input_dim = 6+ 6 + 6

hidden_dim =256
n_layers = 4
linearfc2_dim = 512
linearfc3_dim = 256
output_dim = 54
droupout_rate = 0
recurrent_kind = 'lstm'  # 'rnn' 'gru' 'lstm'


idx_num_epochs = 100
idx_learning_rate =1e-3
idx_weight_decay = 0
scheduler = "cosine_annealing 500"
# scheduler = "step 100 0.1"

# scheduler = "None"


opti_type = 'Adam' 
# opti_type = 'SGD'






# =============================================================================


if operation == 'TRAIN':

    if 'bias' in data_paths[0]:
        idx_dataset_type = '_bias'
    elif 'MEMS' in data_paths[0]:
        idx_dataset_type = '_MEMS'


    if nograd_sd == False:
        idx_dr_grad = '_withgrad_'
    else:
        idx_dr_grad = '_nograd_'    
    if idx_shuffle == False:
        idx_shuffle_str = '_'
    else:
        idx_shuffle_str = 'shuffle_'

    if indx_train_gnssgap:
        inx_train_gnssgap = 'gnssgap_' + str(gnssgap)
    else:
        inx_train_gnssgap = ''

    if idx_ifimu:
        idx_ifimu_str = 'ifimu'
    else:  
        idx_ifimu_str = 'noifimu'

    idx_imu_scale_str = idx_imu_scale[0][0] + idx_imu_scale[0][2:] + idx_imu_scale[1][0] + idx_imu_scale[1][2:] 
 

    if idx_ifatt:
        idx_ifatt_str = 'ifatt'
    else:  
        idx_ifatt_str = 'noifatt'

    idx_att_scale_str = idx_att_scale[0] + idx_att_scale[2:] 

    if idx_feedback_type == 'pvb_':
        idx_ifatt_str = ''
        idx_att_scale_str = ''


    idx_lossweight_coeff_str = ''.join(idx_lossweight_coeff)    
    out_model = today_date  + idx_dataset_type + idx_dr_grad + idx_shuffle_str + inx_train_gnssgap + recurrent_kind + idx_lossweight_coeff_str + '_' + str(idx_num_epochs) + '_' + idx_feedback_type + '_' +  idx_ifimu_str + idx_imu_scale_str + idx_ifatt_str + idx_att_scale_str + '_' + str(target_Fs) + 'Hz_Trajlength' + str(tarj_length) + '_' + idx_other_settings
    out_model_path = currt_folder + '/model/' + out_model + '.pt'

    LoadModel = False
    TrainModel = True
    SaveModel = True
    print(out_model)

elif operation == 'TEST':

    in_model_path = currt_folder + '/model/' + in_model + '.pt'
    print(in_model)

    LoadModel = True
    TrainModel = False
    SaveModel = False

elif operation == 'TRAIN_USEOLD':
    in_model_path = currt_folder + '/model/' + in_model + '.pt'
    print(in_model)
    LoadModel = True
    TrainModel = True
    SaveModel = True
    out_model_path = currt_folder + '/model/' + out_model + '.pt'


elif operation == 'MB_Only':

    LoadModel = False
    TrainModel = False
    SaveModel = False 

# =============================================================================



def get_batch(data, data_fs, tarj_length):
    num_batches = int(data.shape[0] / data_fs // tarj_length)
    data_trimmed  = data[:num_batches*tarj_length*data_fs,:]
    return data_trimmed.reshape(num_batches,tarj_length*data_fs,data.shape[1])

def average_every_Fs(input_tensor,Fs_sample):
    averages = []
    for j in range(input_tensor.shape[1]):
        averages.append([])
        for i in range(0, input_tensor.shape[0], Fs_sample):
            subset = input_tensor[i:i+Fs_sample,j]
            avg = torch.mean(subset)  # 转换为浮点型并计算平均值
            averages[j].append(avg)
        averages[j] = torch.stack(averages[j],dim=0)
    return torch.stack(averages).T


def read_multidata(data_paths, Fs_meas, Fs, tarj_length):


    pos_meas = []
    vel_meas = []
    position_ecef = []
    velocity_ecef = []
    position_ned = []
    velocity_ned = []
    position_llh = []
    orientation_euler_rad = []
    orientation_euler_rad = []
    accbody = []
    angularVelocity = []
    for data_path in data_paths:
        # Adjust the read_csv parameters based on your file format and structure
        df = pd.read_csv(data_path)
        

        # GT (ADMA)
        pos_llh_GT = torch.tensor(df.loc[::GT_samp,['LatRadian','LonRadian','HeightMeters']].dropna().values)
        pos_ecef_GT = torch.tensor(df.loc[::GT_samp,['ECEF_x1','ECEF_y1', 'ECEF_z1']].dropna().values)
        vel_ecef_GT = torch.tensor(df.loc[::GT_samp,['Velocity_X_ECEF_GT','Velocity_Y_ECEF_GT','Velocity_Z_ECEF_GT']].dropna().values)
        pos_ned_GT = torch.tensor(df.loc[::GT_samp,['xNorth_GT','yEast_GT','zDown_GT']].dropna().values)
        vel_ned_GT = torch.tensor(df.loc[::GT_samp,['Velocity_N','Velocity_E','Velocity_D']].dropna().values)
        # imu_GT = torch.tensor(df.loc[::GT_samp,['GT_AccX','GT_AccY','GT_AccZ','GT_RateX','GT_RateY','GT_RateZ']].dropna().values)
        #The rotation manner to be checked
        att_GT = torch.tensor(df.loc[::GT_samp,['RollRad','PitchRad','YawRad']].dropna().values)
        # ATT From ENU RFU --- NED FRD
        # att_GT[:,2] = torch.pi /2 - att_GT[:,2]

        # att_GT[:, 2] = att_GT[:, 2] + 2 * torch.pi * (att_GT[:, 2] < -torch.pi)


        # GNSS
        pos_ecef_GNSS = torch.tensor(df.loc[:,['ECEF_x2','ECEF_y2','ECEF_z2']].dropna().values)
        vel_ecef_GNSS = torch.tensor(df.loc[:,['Velocity_X_ECEF_GNSS','Velocity_Y_ECEF_GNSS','Velocity_Z_ECEF_GNSS']].dropna().values)
        # IMU
        # imu_BNO_PP = torch.tensor(df.loc[::IMU_samp,['LinearX_BNO_Postprocess','LinearY_BNO_Postprocess','LinearZ_BNO_Postprocess','GyroX_BNO_Postprocess', 'GyroY_BNO_Postprocess', 'GyroZ_BNO_Postprocess',]].dropna().values)
        # imu_BNO_PP[:,[1,2,4,5]] = -imu_BNO_PP[:,[1,2,4,5]]
        
        # imu_LSM_PP = torch.tensor(df.loc[::IMU_samp,['LinearX_LSM_Postprocess','LinearY_LSM_Postprocess','LinearZ_LSM_Postprocess','GyroX_LSM_Postprocess', 'GyroY_LSM_Postprocess', 'GyroZ_LSM_Postprocess',]].dropna().values)
        # imu_LSM_PP[:,[1,2,4,5]] = -imu_LSM_PP[:,[1,2,4,5]]
        # imu_FXOS_PP = torch.tensor(df.loc[::IMU_samp,['LinearX_FXOS_Postprocess','LinearY_FXOS_Postprocess','LinearZ_FXOS_Postprocess','GyroX_FXAS_Postprocess', 'GyroY_FXAS_Postprocess', 'GyroZ_FXAS_Postprocess',]].dropna().values)
        # imu_FXOS_PP[:,[1,2,4,5]] = -imu_FXOS_PP[:,[1,2,4,5]]


        # imu_BNO_RAW = torch.tensor(df.loc[::IMU_samp,['LinearX_BNO_Raw','LinearY_BNO_Raw','LinearZ_BNO_Raw','GyroX_BNO_Raw', 'GyroY_BNO_Raw', 'GyroZ_BNO_Raw',]].dropna().values)
        # imu_BNO_RAW[:,[1,2,4,5]] = -imu_BNO_RAW[:,[1,2,4,5]]
        # imu_LSM_RAW = torch.tensor(df.loc[::IMU_samp,['LinearX_LSM_Raw','LinearY_LSM_Raw','LinearZ_LSM_Raw','GyroX_LSM_Raw', 'GyroY_LSM_Raw', 'GyroZ_LSM_Raw',]].dropna().values)
        # imu_LSM_RAW[:,[1,2,4,5]] = -imu_LSM_RAW[:,[1,2,4,5]]
        # imu_FXOS_RAW = torch.tensor(df.loc[::IMU_samp,['LinearX_FXOS_Raw','LinearY_FXOS_Raw','LinearZ_FXOS_Raw','GyroX_FXAS_Raw', 'GyroY_FXAS_Raw', 'GyroZ_FXAS_Raw',]].dropna().values)
        # imu_FXOS_RAW[:,[1,2,4,5]] = -imu_FXOS_RAW[:,[1,2,4,5]]

        imu= torch.tensor(df.loc[::IMU_samp,['AccX_simu','AccY_simu','AccZ_simu','GyroX_simu','GyroY_simu','GyroZ_simu']].dropna().values)
    

        # imu= torch.tensor(df.loc[:,['AccX_simu','AccY_simu','AccZ_simu','GyroX_simu','GyroY_simu','GyroZ_simu']].dropna().values)
        # imu = average_every_Fs(imu,IMU_samp)

        pos_meas.append(get_batch(pos_ecef_GNSS, Fs_meas, tarj_length))
        vel_meas.append(get_batch(vel_ecef_GNSS, Fs_meas, tarj_length))


        # time_traj = torch.from_numpy(dataset[0, 0]["time_traj_real"]).float()
        position_ecef.append(get_batch(pos_ecef_GT, Fs, tarj_length))
        velocity_ecef.append(get_batch(vel_ecef_GT, Fs, tarj_length))
        position_ned.append(get_batch(pos_ned_GT, Fs, tarj_length))
        velocity_ned.append(get_batch(vel_ned_GT, Fs, tarj_length))
        position_llh.append(get_batch(pos_llh_GT, Fs, tarj_length))

        # orientation_euler = torch.from_numpy(dataset[0, 0]["orientation_euler"]).float()
        orientation_euler_rad.append(get_batch(att_GT, Fs, tarj_length))

        accbody.append(get_batch(imu[:,:3], Fs, tarj_length))
        angularVelocity.append(get_batch(imu[:,3:], Fs, tarj_length))

    pos_meas_cat = torch.cat(pos_meas, dim=0)
    vel_meas_cat = torch.cat(vel_meas, dim=0)
    position_ecef_cat = torch.cat(position_ecef, dim=0)
    velocity_ecef_cat = torch.cat(velocity_ecef, dim=0)
    position_ned_cat = torch.cat(position_ned, dim=0)
    velocity_ned_cat = torch.cat(velocity_ned, dim=0)
    position_llh_cat = torch.cat(position_llh, dim=0)
    orientation_euler_rad_cat = torch.cat(orientation_euler_rad, dim=0)
    accbody_cat = torch.cat(accbody, dim=0)
    angularVelocity_cat = torch.cat(angularVelocity, dim=0)


    return pos_meas_cat,vel_meas_cat,position_ecef_cat,velocity_ecef_cat,position_ned_cat, velocity_ned_cat,position_llh_cat,orientation_euler_rad_cat,orientation_euler_rad_cat,accbody_cat,angularVelocity_cat

pos_meas,vel_meas,position_ecef,velocity_ecef,position_ned, velocity_ned,position_llh,orientation_euler_rad,orientation_euler_rad,accbody,angularVelocity = read_multidata(data_paths, Fs_meas, Fs, tarj_length)


acceleration = torch.zeros(position_llh.shape[0],position_llh.shape[1],3)

train_num = round(pos_meas.shape[0]*0.9)
test_num = pos_meas.shape[0] - train_num

att_ecef_euler_rad = torch.zeros(orientation_euler_rad.shape[0],orientation_euler_rad.shape[1],3)
reshape_C_b_e = torch.zeros(orientation_euler_rad.shape[0],orientation_euler_rad.shape[1],9)
for ii in range(orientation_euler_rad.shape[0]):
    for jj in range(orientation_euler_rad.shape[1]):
        est_C_b_n = functions.euler_to_CTM(orientation_euler_rad[ii,jj]).T
        _,_,est_C_b_e= functions.geo_ned2ecef_0(position_llh[ii,jj], velocity_ned[ii,jj], est_C_b_n)
        att_ecef_euler_rad[ii,jj] = functions.CTM_to_euler(est_C_b_e.T)
        reshape_C_b_e[ii,jj] = est_C_b_e.reshape(9)
# time_meas = torch.from_numpy(dataset[0, 0]["time_meas"]).float()
# rangepos1 = torch.from_numpy(dataset[0, 0]['rangepos1'].astype(datatype))
# rangepos2 = torch.from_numpy(dataset[0, 0]['rangepos2'].astype(datatype))
# rangepos3 = torch.from_numpy(dataset[0, 0]['rangepos3'].astype(datatype))
# allref = torch.cat((rangepos1, rangepos2, rangepos3), dim=1).to(dev)
# rangeacc = float(dataset[0, 0]["rangeacc"])
# r1 = torch.unsqueeze((torch.from_numpy(dataset[0, 0]["r1"].astype(datatype))), dim=-1)
# r2 = torch.unsqueeze((torch.from_numpy(dataset[0, 0]["r2"].astype(datatype))), dim=-1)
# r3 = torch.unsqueeze((torch.from_numpy(dataset[0, 0]["r3"].astype(datatype))), dim=-1)

# b_a = torch.from_numpy(dataset[0, 0]['b_a'].astype(datatype))[0][0]
# b_g = torch.from_numpy(dataset[0, 0]['b_g'].astype(datatype))[0][0]

# pos_meas =  torch.from_numpy(dataset[0, 0]['pos_meas'].astype(datatype))

# =============================================================================
#                   Step2 Using PyTorch Dataset and DataLoader
# =============================================================================
# Linear Dataset
# X = torch.cat((r1, r2, r3), dim=2).to(dev)
X = torch.cat((pos_meas, vel_meas), dim=2).to(dev)
T = torch.cat((position_ecef, velocity_ecef, position_ned, velocity_ned, position_llh, acceleration, orientation_euler_rad, att_ecef_euler_rad, reshape_C_b_e), dim=2).to(dev)
IMU = torch.cat((accbody, angularVelocity), dim=2).to(dev)
IMU = IMU.reshape(IMU.shape[0], -1, Fs, 6)


# shuffle
idx_s = torch.randperm(X.shape[0])
X = X[idx_s].view(X.size())
T = T[idx_s].view(T.size())
IMU = IMU[idx_s].view(IMU.size())


# Circular Dataset
# X = torch.from_numpy(np.load("data/rangemeas_circle.npy")).to(dev)
# T = torch.from_numpy(np.load("data/trajectory_ref_circle.npy")).to(dev)

# Mix Dataset
# X = torch.from_numpy(np.load("data/rangemeas_Mix_4d.npy")).to(dev)
# T = torch.from_numpy(np.load("data/trajectory_ref_Mix_4d.npy")).to(dev)
# train_num=20
''' the train batch could be larger'''
train_batch_size = idx_train_batch_size  # choose batch size
train_val_splitter = 0.9

train_features = X[0 : int(train_num * train_val_splitter)].to(dev)
train_IMU = IMU[0 : int(train_num * train_val_splitter)].to(dev)
train_targets = T[0 : int(train_num * train_val_splitter)].to(dev)
# train_targets = T2[0:trainsets_num]

########### Test long train traj
# train_features = train_features.reshape(9,-1,6)
# train_IMU = train_IMU.reshape(9,-1,10,6)
# train_targets = train_targets.reshape(9,-1,33)



val_features = X[int(train_num * train_val_splitter) : train_num].to(dev)
val_IMU = IMU[int(train_num * train_val_splitter) : train_num].to(dev)
val_targets = T[int(train_num * train_val_splitter) : train_num].to(dev)

# val_features = train_features
# val_targets = train_targets

# 20s traj to test
# test_time_traj = time_traj[train_num:].to(dev)
test_features = X[train_num:].to(dev)
test_IMU = IMU[train_num:].to(dev)
test_targets = T[train_num:].to(dev)
# test_targets = T2[trainsets_num:]

# Longer test traj
# indx_test = idx_cut_test_traj
# # test_time_traj = test_time_traj.reshape(indx_test,-1)
# test_features = train_features.reshape(indx_test,-1,6)
# test_IMU = train_IMU.reshape(indx_test,-1,Fs,6)
# test_targets = train_targets.reshape(indx_test,-1,33)


# Longer test traj
# indx_test = idx_cut_test_traj
# # test_time_traj = test_time_traj.reshape(indx_test,-1)
# test_features = test_features.reshape(indx_test,-1,6)
# test_IMU = test_IMU.reshape(indx_test,-1,Fs,6)
# test_targets = test_targets.reshape(indx_test,-1,33)

print("data loaded")


class Range_INS_Dataset(Dataset):
    def __init__(self, features, targets, imu_meas):
        self.features = features
        self.targets = targets  
        self.imu_meas = imu_meas
        self.num_traj = features.shape[0]
        self.time_step = features.shape[1]

    def __getitem__(self, index):
        return self.features[index], self.targets[index], self.imu_meas[index]

    def __len__(self):
        return self.num_traj


train_dataset = Range_INS_Dataset(train_features, train_targets, train_IMU)

if idx_shuffle == True:
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=train_batch_size, shuffle=True
    )
else:
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=train_batch_size, shuffle=False
    )  

'''Here, if shuffle = false, we only take the fisrt train_batch_size trajectory to train'''

val_dataset = Range_INS_Dataset(val_features, val_targets, val_IMU)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)

test_dataset = Range_INS_Dataset(test_features, test_targets, test_IMU)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

# =============================================================================
#                               Init Net
# =============================================================================
"""est_P"""

# m = 9
# # n = 6

# input_dim = m
# linearfc1_dim = m * 50
# hidden_dim = (m**2) * 10 * 1
# n_layers = 2
# linearfc2_dim = m * 20
# output_dim = m * m

# net = Rnn_Kalman_Model.LC_est_P(
#     input_dim, linearfc1_dim, hidden_dim, n_layers, linearfc2_dim, output_dim
# )
# # not really necessary since default tensor type is already set appropriately
# net.to(dev)
# net = torch.load("test_P.pt", map_location=dev)
"""est_P_Q_R"""

# m = 9
# n = 6

# input_dim_1 = m 
# linearfc1_dim_1 = m * 100
# hidden_dim_1 = (m ** 2) * 10
# n_layers_1 = 16
# linearfc2_dim_1 = m * 20
# output_dim_1 = m * m

# input_dim_2 = m
# linearfc1_dim_2 = m * 100
# hidden_dim_2 = (m ** 2) * 10
# n_layers_2 = 1
# linearfc2_dim_2 = m * 20
# output_dim_2 = m * m

# input_dim_3 = n
# linearfc1_dim_3 = n * 50
# hidden_dim_3 = (n ** 2) * 10
# n_layers_3 = 1
# linearfc2_dim_3 = n * 20
# output_dim_3 = n * n

# net = Rnn_Kalman_Model.LC_est_P_Q_R(input_dim_1, linearfc1_dim_1, hidden_dim_1, n_layers_1, linearfc2_dim_1, output_dim_1,
#                                     input_dim_2, linearfc1_dim_2, hidden_dim_2, n_layers_2, linearfc2_dim_2, output_dim_2,
#                                     input_dim_3, linearfc1_dim_3, hidden_dim_3, n_layers_3, linearfc2_dim_3, output_dim_3)

# net = torch.load('test_Q_P_R.pt', map_location=dev)
"""est_KG"""

net = RnnModel.LC_est_KG(
    dev,
    input_dim,
    hidden_dim,
    n_layers,
    linearfc2_dim,
    linearfc3_dim,
    output_dim,
    droupout_rate,
    recurrent_kind,
    Fs,
    idx_feedback_type,
    idx_ifimu,
    idx_ifatt,
    idx_imu_scale,
    idx_att_scale,
    idx_train_batch_size,
)
net.to(dev)

if LoadModel:
    net.load_state_dict(torch.load(in_model_path))

# =============================================================================
#                           Step4 Train the model
# =============================================================================

def make_scheduler(scheduler):
    if scheduler == "None":
        return None
    scheduler_kind, *scheduler_params = scheduler.split(" ")
    scheduler_params = [float(x) for x in scheduler_params]
    gen = None
    if scheduler_kind == "cosine_annealing":
        gen = CosineAnnealingLR
    elif scheduler_kind == "cosine_annealing_warm":
        gen = CosineAnnealingWarmRestarts
    elif scheduler_kind == "step":
        gen = StepLR
    else:
        raise ValueError("Invalid scheduler")
    return lambda optimizer: gen(optimizer, *scheduler_params)


scheduler = "cosine_annealing 500"
# scheduler = "step 100 0.1"

# scheduler = "None"

Trainer = Pipeline.Pipeline_LC(
    net,
    dev,
    num_epochs=idx_num_epochs,
    learning_rate=idx_learning_rate,
    weight_decay=idx_weight_decay,
    loss_fn=nn.MSELoss(reduction="mean"),
    scheduler_generator=make_scheduler(scheduler),
    nograd = nograd_sd,
    lossweight_coeff = idx_lossweight_coeff,
    train_gnssgap = indx_train_gnssgap,
    gnssgap = gnssgap,
    opti_type = opti_type,
)
# use imu_time_interval = 1 for fast training, but increase error in dr, imu frequency is 100Hz, use all of it will be super slow
# Change float32 to 64 is very imporant when use time interval smaller that 1s. Accuary of DR changes from 70 to 0.001!

if TrainModel:
    timer = functions.Timer()
    Trainer.train_lc(train_loader, train_dataset, val_loader, val_dataset, Fs, Fs_meas)
    print(f"{timer.stop():.2f} sec")

if SaveModel:
    torch.save(net.state_dict(), out_model_path)


# =============================================================================
#                               Test
# =============================================================================
# todo: change to a test dataset
# this is fine as long as we don't use the validation dataset (i.e. as a stopping criterion)
# Change output to checklist to check several params
est_traj_nn, ref_traj, bias_history, est_traj_nn_llh, ref_traj_llh, predict_KG_net= Trainer.test_lc(test_loader, test_dataset, Fs, Fs_meas)

# KGain = check_list_test[0].cpu().detach().numpy()
# P = check_list_test[1].cpu().detach().numpy()
# Q = check_list_test[2].cpu().detach().numpy()
# R = check_list_test[3].cpu().detach().numpy()
# est_IMU_bias_new_array = est_IMU_bias_new.cpu().detach().numpy()

# est_att_ned_euler_rad = torch.zeros(est_traj_nn.shape[0],est_traj_nn.shape[1],3)
# for ii in range(est_traj_nn.shape[0]):
#     for jj in range(est_traj_nn.shape[1]):
#         est_C_b_e = functions.euler_to_CTM(est_traj_nn[ii,jj,6:]).T
#         _,_,est_att_ned_euler_rad[ii,jj]= functions.ecef2geo_ned(est_traj_nn[ii,jj,:3], est_traj_nn[ii,jj,3:6], est_C_b_e)


# # =============================================================================
# #                           Re produce the Covariance
# # =============================================================================

# H = torch.zeros((6, 9))
# H[0:3, 0:3] = -torch.eye(3)
# H[3:6, 3:6] = -torch.eye(3)

# predict_Cov_from_KG_net = functions.KG2COV(predict_KG_net, H)





# =============================================================================
#                              Position Accuracy
# =============================================================================

def rem2pi(tensor: torch.Tensor) -> torch.Tensor:
    return torch.remainder(tensor, 2 * np.pi)


def loss(predicted: torch.Tensor, reference: torch.Tensor) -> float:
    return (torch.sum(torch.norm(predicted - reference, dim=1)) / reference.shape[1]).item()


def loss_total(start: int, end: int, predicted: torch.Tensor,
               transform: Callable[[torch.Tensor], torch.Tensor] = lambda x: x) -> float:
    return loss(transform(predicted[:, :, start:end]), transform(ref_traj[:, :, start:end]))


losses = {
    "position": {
        "lstm": loss_total(0, 3, est_traj_nn),
    },
    "velocity": {
        "lstm": loss_total(3, 6, est_traj_nn),
    },
    # "attitude": {
    #     "lstm": loss_total(4, 5, est_traj_nn, rem2pi),
    # }
}

print("Losses:")
pprint(losses)


# =============================================================================
#                                     Model Based
# =============================================================================


if indx_MB:

    MB_dataset_type = 'real'
    est_traj_nn_MB, ref_traj_MB, bias_history_MB, est_traj_nn_llh_MB, ref_traj_llh_MB, P_MB, KG_MB= Trainer.GnssInsLooseCoupling(MB_dataset_type, test_loader, test_dataset, Fs, Fs_meas)


    losses = {
    "position": {
        "lstm": loss_total(0, 3, est_traj_nn_MB),
    },
    "velocity": {
        "lstm": loss_total(3, 6, est_traj_nn_MB),
    },
    # "attitude": {
    #     "lstm": loss_total(4, 5, est_traj_nn, rem2pi),
        # }
    }

    print("MB Losses:")
    pprint(losses)


# =============================================================================
#                                     Plot
# =============================================================================

est_traj_nn = est_traj_nn.detach().cpu().numpy()
est_traj_nn_MB = est_traj_nn_MB.detach().cpu().numpy()
ref_traj = ref_traj.detach().cpu().numpy()
bias_history = bias_history.detach().cpu().numpy()
est_traj_nn_llh = est_traj_nn_llh.numpy()
est_traj_nn_llh_MB = est_traj_nn_llh_MB.numpy()
ref_traj_llh = ref_traj_llh.numpy()
predict_KG_net = predict_KG_net.detach().cpu().numpy()
P_MB = P_MB.detach().cpu().numpy()
KG_MB = KG_MB.detach().cpu().numpy()


sumloss_p = 0
sumloss_v = 0
sumloss_p_MB = 0
sumloss_v_MB = 0
for i in range(est_traj_nn.shape[1]):
    sumloss_p = sumloss_p + ((est_traj_nn[0,i,0] - ref_traj[0,i,0])**2 + (est_traj_nn[0,i,1] - ref_traj[0,i,1])**2 + (est_traj_nn[0,i,2] - ref_traj[0,i,2])**2)**0.5
    sumloss_v = sumloss_v + ((est_traj_nn[0,i,3] - ref_traj[0,i,3])**2 + (est_traj_nn[0,i,4] - ref_traj[0,i,4])**2 + (est_traj_nn[0,i,5] - ref_traj[0,i,5])**2)**0.5
    sumloss_p_MB = sumloss_p_MB + ((est_traj_nn_MB[0,i,0] - ref_traj[0,i,0])**2 + (est_traj_nn_MB[0,i,1] - ref_traj[0,i,1])**2 + (est_traj_nn_MB[0,i,2] - ref_traj[0,i,2])**2)**0.5
    sumloss_v_MB = sumloss_v_MB + ((est_traj_nn_MB[0,i,3] - ref_traj[0,i,3])**2 + (est_traj_nn_MB[0,i,4] - ref_traj[0,i,4])**2 + (est_traj_nn_MB[0,i,5] - ref_traj[0,i,5])**2)**0.5
meanloss_p = sumloss_p/est_traj_nn.shape[1]
meanloss_v = sumloss_v/est_traj_nn.shape[1]
sumloss_p_MB = sumloss_p_MB/est_traj_nn.shape[1]
sumloss_v_MB = sumloss_v_MB/est_traj_nn.shape[1]

# Pos LSTM
plt.figure()
plt.grid()
plt.title("Net position")
for k in range(ref_traj_llh.shape[0]):  # est_traj_nn.shape[0]):
    plt.plot(ref_traj_llh[k, :, 1], ref_traj_llh[k, :, 0], "b", label="Reference" if k == 0 else None)
    plt.plot(est_traj_nn_llh_MB[k, :, 1], est_traj_nn_llh_MB[k, :, 0], "g", label="MB" if k == 0 else None)
    plt.plot(est_traj_nn_llh[k, :, 1], est_traj_nn_llh[k, :, 0], "r", label="Net" if k == 0 else None)
    
plt.legend()

# Pos LSTM
plt.figure()
plt.grid()
plt.title("Net position Height")
for k in range(ref_traj_llh.shape[0]):  # est_traj_nn.shape[0]):
    plt.plot(ref_traj_llh[k, :, 2], "b", label="Reference"  if k == 0 else None)
    plt.plot(est_traj_nn_llh_MB[k, :, 2], "g", label="MB"  if k == 0 else None)
    plt.plot(est_traj_nn_llh[k, :, 2], "r", label="Net"  if k == 0 else None)
plt.legend()


# Pos LSTM
# plt.figure()
# plt.grid()
# plt.title("Net position")
# for k in range(5):  # est_traj_nn.shape[0]):
#     plt.plot(ref_traj[k, :, 1], ref_traj[k, :, 0], "b", label="Reference")
#     plt.plot(est_traj_nn[k, :, 1], est_traj_nn[k, :, 0], "r", label="Net")
# plt.legend()


plt.figure()
plt.grid()
plt.title("Attitude")
for k in range(ref_traj_llh.shape[0]):  # est_traj_nn.shape[0]):
    plt.plot(ref_traj[k, :, 21], "b", label="Reference x"  if k == 0 else None)
    plt.plot(ref_traj[k, :, 22], "b", label="Reference y"  if k == 0 else None)
    plt.plot(ref_traj[k, :, 23], "b", label="Reference z"  if k == 0 else None)
    plt.plot(est_traj_nn[k, :, 6], "r", label="Net x"  if k == 0 else None)
    plt.plot(est_traj_nn[k, :, 7], "r", label="Net y"  if k == 0 else None)
    plt.plot(est_traj_nn[k, :, 8], "r", label="Net z"  if k == 0 else None)
    plt.plot(est_traj_nn_MB[k, :, 6], "g", label="MB"  if k == 0 else None)
    plt.plot(est_traj_nn_MB[k, :, 7], "g", label="MB"  if k == 0 else None)
    plt.plot(est_traj_nn_MB[k, :, 8], "g", label="MB"  if k == 0 else None)

plt.legend()


# def rem2pi_np(array):
#     return np.remainder(array, 2 * np.pi)

plt.figure()
plt.grid()
plt.title("Velocity")
for k in range(ref_traj_llh.shape[0]):  # est_traj_nn.shape[0]):
    plt.plot(est_traj_nn[k, :, 3], "r", label="Net x"  if k == 0 else None)
    plt.plot(est_traj_nn[k, :, 4], "r", label="Net y"  if k == 0 else None)
    plt.plot(est_traj_nn[k, :, 5], "r", label="Net z"  if k == 0 else None)
    plt.plot(ref_traj[k, :, 3], "b", label="Reference x"  if k == 0 else None)
    plt.plot(ref_traj[k, :, 4], "b", label="Reference y"  if k == 0 else None)
    plt.plot(ref_traj[k, :, 5], "b", label="Reference z"  if k == 0 else None)
plt.legend()

plt.figure()
plt.grid()
plt.title("Velocity MB")
for k in range(ref_traj_llh.shape[0]):  # est_traj_nn.shape[0]):
    plt.plot(est_traj_nn_MB[k, :, 3], "r", label="Net x"  if k == 0 else None)
    plt.plot(est_traj_nn_MB[k, :, 4], "r", label="Net y"  if k == 0 else None)
    plt.plot(est_traj_nn_MB[k, :, 5], "r", label="Net z"  if k == 0 else None)
    plt.plot(ref_traj[k, :, 3], "b", label="Reference x"  if k == 0 else None)
    plt.plot(ref_traj[k, :, 4], "b", label="Reference y"  if k == 0 else None)
    plt.plot(ref_traj[k, :, 5], "b", label="Reference z"  if k == 0 else None)
plt.legend()



# est_att_ned_euler_rad = est_att_ned_euler_rad.numpy()

# plt.figure()
# plt.grid()
# plt.title("Attitude")
# for k in range(1):  # est_traj_nn.shape[0]):
#     plt.plot(ref_traj[k, :, 18], "b", label="Reference x")
#     plt.plot(ref_traj[k, :, 19], "b", label="Reference y")
#     plt.plot(ref_traj[k, :, 20], "b", label="Reference z")

# plt.legend()

# plt.figure()
# plt.grid()
# plt.title("Attitude")
# for k in range(1):  # est_traj_nn.shape[0]):
#     plt.plot(est_att_ned_euler_rad[k, :, 0], "r", label="Net x")
#     plt.plot(est_att_ned_euler_rad[k, :, 1], "r", label="Net y")
#     plt.plot(est_att_ned_euler_rad[k, :, 2], "r", label="Net z")

# plt.legend()



# Pos LSTM
# Vel LSTM
# plt.figure()
# plt.grid()
# plt.title("Net velocity")
# for k in range(1):  # est_traj_nn.shape[0]):
#     plt.plot(est_traj_nn_llh[k, ::10, 3], label=r"Net $v_n$")
#     plt.plot(est_traj_nn_llh[k, ::10, 4], label=r"Net $v_e$")
#     plt.plot(est_traj_nn_llh[k, ::10, 5], label=r"Net $v_d$")
#     plt.plot(ref_traj_llh[k, ::10, 3], label=r"Reference $v_n$")
#     plt.plot(ref_traj_llh[k, ::10, 4], label=r"Reference $v_e$")
#     plt.plot(ref_traj_llh[k, ::10, 5], label=r"Reference $v_d$")

# plt.legend()



# Bias history
plt.figure()
plt.grid()
plt.title("Acc")
for k in range(ref_traj_llh.shape[0]):
    plt.plot(bias_history[k, :, 0], "b", label=r"$Acc x$" if k == 0 else None)
    plt.plot(bias_history[k, :, 1], "r", label=r"$Acc y$" if k == 0 else None)
    plt.plot(bias_history[k, :, 2], "g", label=r"$Acc z$" if k == 0 else None)
# plt.plot([0,100],[b_a[0],b_a[0]], label="Reference ba_x")
# plt.plot([0,100],[b_a[1],b_a[1]], label="Reference ba_y")
# plt.plot([0,100],[b_g[2],b_g[2]], label="Reference bg")
plt.legend()

# Bias history
plt.figure()
plt.grid()
plt.title("Gyro")
for k in range(ref_traj_llh.shape[0]):
    plt.plot(bias_history[k, :, 3], "b", label=r"$Gyro x$" if k == 0 else None)
    plt.plot(bias_history[k, :, 4], "r", label=r"$Gyro y$" if k == 0 else None)
    plt.plot(bias_history[k, :, 5], "g", label=r"$Gyro z$" if k == 0 else None)
# plt.plot([0,100],[b_a[0],b_a[0]], label="Reference ba_x")
# plt.plot([0,100],[b_a[1],b_a[1]], label="Reference ba_y")
# plt.plot([0,100],[b_g[2],b_g[2]], label="Reference bg")
plt.legend()

# Bias history
plt.figure()
plt.grid()
plt.title("Acc MB")
for k in range(ref_traj_llh.shape[0]):
    plt.plot(bias_history_MB[k, :, 0], "b", label=r"$Acc x$" if k == 0 else None)
    plt.plot(bias_history_MB[k, :, 1], "r", label=r"$Acc y$" if k == 0 else None)
    plt.plot(bias_history_MB[k, :, 2], "g", label=r"$Acc z$" if k == 0 else None)
# plt.plot([0,100],[b_a[0],b_a[0]], label="Reference ba_x")
# plt.plot([0,100],[b_a[1],b_a[1]], label="Reference ba_y")
# plt.plot([0,100],[b_g[2],b_g[2]], label="Reference bg")
plt.legend()

# Bias history
plt.figure()
plt.grid()
plt.title("Gyro MB")
for k in range(ref_traj_llh.shape[0]):
    plt.plot(bias_history_MB[k, :, 3], "b", label=r"$Gyro x$" if k == 0 else None)
    plt.plot(bias_history_MB[k, :, 4], "r", label=r"$Gyro y$" if k == 0 else None)
    plt.plot(bias_history_MB[k, :, 5], "g", label=r"$Gyro z$" if k == 0 else None)
# plt.plot([0,100],[b_a[0],b_a[0]], label="Reference ba_x")
# plt.plot([0,100],[b_a[1],b_a[1]], label="Reference ba_y")
# plt.plot([0,100],[b_g[2],b_g[2]], label="Reference bg")
plt.legend()


plt.figure()
plt.grid()
plt.title("C_b_e")
for k in range(ref_traj_llh.shape[0]):  # est_traj_nn.shape[0]):
    plt.plot(ref_traj[k, :, 23], "b", label="ref"  if k == 0 else None)
    plt.plot(est_traj_nn_MB[k, :, 8], "g", label="MB"  if k == 0 else None)
    plt.plot(est_traj_nn[k, :, 8], "r", label="Net"  if k == 0 else None)
plt.legend()

if save_to_matlab and operation == 'TEST':
    scio.savemat(
        out_mat_file,
        mdict={
            "ref_traj": ref_traj,
            "est_traj_nn": est_traj_nn,
            "est_traj_nn_MB": est_traj_nn_MB,
            "ref_traj_llh": ref_traj_llh,
            "est_traj_nn_llh": est_traj_nn_llh,
            "est_traj_nn_llh_MB": est_traj_nn_llh_MB,
            "est_KG_nn":predict_KG_net,
            "meanloss_p": meanloss_p,
            "meanloss_v": meanloss_v,
            # "time_traj" : test_time_traj,
        },
    )

# scio.savemat(
#     out_mat_file,
#     mdict={
#         "ref_traj": ref_traj,
#         "est_traj_nn": est_traj_nn,
#         "est_traj_nn_MB": est_traj_nn_MB,
#         "ref_traj_llh": ref_traj_llh,
#         "est_traj_nn_llh": est_traj_nn_llh,
#         "est_traj_nn_llh_MB": est_traj_nn_llh_MB,
#         "est_KG_nn":predict_KG_net,
#         "P_MB": P_MB,
#         "KG_MB": KG_MB,
#         # "time_traj" : test_time_traj,
#     },
# )


# Calculate Position Error
# pos_err_NN_test = np.zeros(est_pos_test_array.shape[0])
# pos_err_LC_test = np.zeros(lc_array_cut.shape[0])
# for i in range(est_pos_test_array.shape[0]):
#     pos_err_NN_test[i] = (
#         (ref_array_cut[i, 1] - est_pos_test_array[i, 0]) ** 2
#         + (ref_array_cut[i, 2] - est_pos_test_array[i, 1]) ** 2
#         + (ref_array_cut[i, 3] - est_pos_test_array[i, 2]) ** 2
#     ) ** (1 / 2)
#     pos_err_LC_test[i] = (
#         (ref_array_cut[i, 1] - lc_array_cut[i, 1]) ** 2
#         + (ref_array_cut[i, 2] - lc_array_cut[i, 2]) ** 2
#         + (ref_array_cut[i, 3] - lc_array_cut[i, 3]) ** 2
#     ) ** (1 / 2)

# # Calculate LLH and Vel NED of test result
# est_pos_test_array_llh = np.zeros((est_pos_test_array.shape[0], 3))
# est_vel_test_array_ned = np.zeros((est_pos_test_array.shape[0], 3))
# for i in range(est_pos_test_array.shape[0]):
#     pos_llh, vel_ned = rnm.ecef2geo_ned_array(
#         est_pos_test_array[i, 0:3], est_vel_test_array[i, 0:3]
#     )
#     est_pos_test_array_llh[i, 0:3] = pos_llh.reshape(3)
#     est_vel_test_array_ned[i, 0:3] = vel_ned.reshape(3)

# # Plot Postioning Error
# rnm.plt.figure()
# rnm.plot([pos_err_NN_test, pos_err_LC_test], legend=("NN", "LC"), title='Position Error')
# # Plot Traj 
# rnm.plt.figure()
# rnm.plot(
#     [est_pos_test_array_llh[:, 1], lc_array_cut[:, 8], ref_array_cut[0:, 8]],
#     [est_pos_test_array_llh[0:, 0], lc_array_cut[:, 7], ref_array_cut[0:, 7]],
#     legend=("NN", "LC", "Ref"), title='Trajectory'
# )
# # Plot Velocity NED
# rnm.plt.figure()
# rnm.plot(
#     [
#         est_vel_test_array_ned[:, 0],
#         est_vel_test_array_ned[:, 1],
#         est_vel_test_array_ned[:, 2],
#     ], title='Velocity NED'
# )
# # Plot Attitude 
# rnm.plt.figure()
# # rnm.plot([est_att_test_array[:,0],lc_array_cut[:,13],ref_array_cut[0:,13],
# #           est_att_test_array[:,1],lc_array_cut[:,14],ref_array_cut[0:,14],
# #           est_att_test_array[:,2],lc_array_cut[:,15],ref_array_cut[0:,15]], legend=('NN','LC','Ref'))
# rnm.plot([est_att_test_array[:,2],lc_array_cut[:,15],ref_array_cut[0:,15]], legend=('NN','LC','Ref'))
# # Plot Acc bias and Gyro Bias
# rnm.plt.figure()
# rnm.plot([est_IMU_bias_new_array[:,0],est_IMU_bias_new_array[:,1],est_IMU_bias_new_array[:,2]])

plt.show()  # Plots don't show up without this on my machine, feel free to remove it
# # Pos LSTM
# plt.figure()
# plt.grid()
# plt.title("Net position Height")
# for k in range(1):  # est_traj_nn.shape[0]):
#     plt.plot(ref_traj_llh[k, :, 2], "b", label="Reference" if k == 0 else None)
#     plt.plot(est_traj_nn_llh[k, :, 2], "r", label="Net" if k == 0 else None)
# plt.legend()
