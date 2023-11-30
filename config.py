import os
batch_size = 16
use_gpu = True
img_size = 448  # unet, resunet50, swinunet, GTNet, GATrans
# img_size = 512  # attentionunet
overlap = 32
epoches = 2000
base_lr = 0.001
weight_decay = 2e-5
momentum = 0.9
power = 0.99
gpu_id = '0'
loss_type = 'ce'
save_iter = 1
num_workers = 1
val_visual = True
image_driver = 'pillow' # pillow, gdal
num_class = 6
model_name = 'Unet'  # ResUNet50, AttentionUnet, SwinUNet, GTNet, GATrans
input_bands = 3
if input_bands == 4:
    mean = (0.472455, 0.320782, 0.318403, 0.357)
    std = (0.144, 0.151, 0.211, 0.195)
else:
    mean = (0.472455, 0.320782, 0.318403)
    std = (0.215084, 0.408135, 0.409993)  # 标准化参数

root_data = '../Vaihingen/cut_data'
dataset = 'massroad'
exp_name = 'model_name'
gen_save_dir = '../gen_{}_files'.format(exp_name)
generator_dir = os.path.join(gen_save_dir, './pth_{}/'.format(model_name))
dis_save_dir = '../dis_{}_files'.format(exp_name)
discrimintor_dir = os.path.join(dis_save_dir, './pth_{}/'.format(model_name))
model_experision = '1'
save_dir = '../{}_files'.format(exp_name)
if os.path.exists(save_dir) is False:
    os.mkdir(save_dir)
save_dir_model = os.path.join(save_dir, model_name+'_'+model_experision)
if os.path.exists(save_dir_model) is False:
    os.mkdir(save_dir_model)
model_dir = os.path.join(save_dir_model, './pth_{}/'.format(model_name))
if os.path.exists(gen_save_dir) is False:
    os.mkdir(gen_save_dir)
if os.path.exists(dis_save_dir) is False:
    os.mkdir(dis_save_dir)
data_dir = os.path.join(save_dir, 'data_slice_{}'.format(dataset))
train_path = os.path.join(root_data, 'train')
train_gt = os.path.join(root_data, 'train_labels')
val_path = '../Vaihingen/train_img/val'
val_gt = '../Vaihingen/val_gt'
test_path = '../Vaihingen/train_img/val'
test_gt = os.path.join(root_data, 'test_labels')
save_path = '../model_name_results'

# save path
# output = os.path.join(save_dir_model, './result_{}/'.format(model_name))
# output_gray = os.path.join(save_dir_model, './result_gray_{}/'.format(model_name))

# model2_name = 'dis_unet'
# save_dir2 = '../{}_files'.format(model2_name)
# save_discrimintor_model = os.path.join(save_dir2, model2_name+'_'+model_experision)
# model2_dir = os.path.join(save_discrimintor_model, './pth_{}/'.format(model2_name))