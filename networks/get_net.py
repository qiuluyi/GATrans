from networks import *

def get_net(model_name, input_bands, num_class):

    if model_name == 'Unet':
        net = unet(input_bands=input_bands, n_classes=num_class, thread_pro=0.4)
    elif model_name == 'ResUNet50':
        net = Res_UNet_50(input_bands, num_class, thread_pro=0.4)
    elif model_name == 'AttentionUnet':
        net = unet_att(input_bands=input_bands, n_classes=num_class, thread_pro=0.4)
    elif model_name == 'SwinUNet':
        net = SwinTransformer()
    elif model_name == 'GTNet':
        net = SwinTransformer()
    elif model_name == 'GATrans':
        net = Generator()
        discrimintor = Discriminator()
        return net, discrimintor
    else:
        raise ('this model is not exist!!!!')

    return net
