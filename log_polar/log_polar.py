import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import cv2
import math
from torch.autograd.gradcheck import gradcheck
from torchvision import transforms, utils
import matplotlib.pyplot as plt

def polar_transformer(U, out_size, device, log=True, radius_factor=0.707):
    """Polar Transformer Layer

    Based on https://github.com/tensorflow/models/blob/master/transformer/spatial_transformer.py.
    _repeat(), _interpolate() are exactly the same;
    the polar transform implementation is in _transform()

    Args:
        U, theta, out_size, name: same as spatial_transformer.py
        log (bool): log-polar if True; else linear polar
        radius_factor (float): 2maxR / Width
    """
    def _repeat(x, n_repeats):
        rep = torch.ones(n_repeats)
        rep.unsqueeze(0)
        x = torch.reshape(x, (-1, 1))
        x = x * rep
        return torch.reshape(x, [-1])

    def _interpolate(im, x, y, out_size): # im [B,H,W,C]
        # constants
        x = x.to(device)
        y = y.to(device)
        num_batch = im.shape[0]
        height = im.shape[1]
        width = im.shape[2]
        channels = im.shape[3]
        height_f = height
        width_f = width
        
        x = x.double()
        y = y.double()
        out_height = out_size[0]
        out_width = out_size[1]
        zero = torch.zeros([])
        max_y = im.shape[1] - 1
        max_x = im.shape[2] - 1

        # do sampling
        x0 = torch.floor(x).long()
        x1 = x0 + 1
        y0 = torch.floor(y).long()
        y1 = y0 + 1

        x0 = torch.clamp(x0, zero, max_x)
        x1 = torch.clamp(x1, zero, max_x)
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)
        dim2 = width
        dim1 = width*height

        base = _repeat(torch.range(0, num_batch-1, dtype=int)*dim1, out_height*out_width)
        base = base.long()
        base = base.to(device)
        base_y0 = base + y0*dim2
        base_y1 = base + y1*dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1
        
        
        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = torch.reshape(im, [-1, channels])
        im_flat = im_flat.clone().float().to(device)

        Ia = im_flat.gather(0, idx_a.unsqueeze(1))
        Ib = im_flat.gather(0, idx_b.unsqueeze(1))
        Ic = im_flat.gather(0, idx_c.unsqueeze(1))
        Id = im_flat.gather(0, idx_d.unsqueeze(1))

        # Ia = im_flat[idx_a].to(device)
        # Ib = im_flat[idx_b].to(device)
        # Ic = im_flat[idx_c].to(device)
        # Id = im_flat[idx_d].to(device)

        # and finally calculate interpolated values
        x0_f = x0.double()
        x1_f = x1.double()
        y0_f = y0.double()
        y1_f = y1.double()
        # print(((x1_f-x) * (y1_f-y)).shape)
        # print("-------------")
        wa = ((x1_f-x) * (y1_f-y)).unsqueeze(1)
        wb = ((x1_f-x) * (y-y0_f)).unsqueeze(1)
        wc = ((x-x0_f) * (y1_f-y)).unsqueeze(1)
        wd = ((x-x0_f) * (y-y0_f)).unsqueeze(1)

        # output = Ia + Ib + Ic + Id
        output = wa*Ia + wb*Ib + wc*Ic + wd*Id
        return output

    def _meshgrid(height, width):
        x_t = torch.ones([height, 1]) * torch.linspace(0.0, 1.0 * width-1, width).unsqueeze(1).permute(1, 0)
        y_t = torch.linspace(0.0, 1.0, height).unsqueeze(1) * torch.ones([1, width])

        x_t_flat = torch.reshape(x_t, (1, -1))
        y_t_flat = torch.reshape(y_t, (1, -1))
        grid = torch.cat((x_t_flat, y_t_flat), 0)

        return grid

    def _transform(input_dim, out_size):
        # radius_factor = torch.sqrt(torch.tensor(2.))/2.
        num_batch = input_dim.shape[0]  # input [B,H,W,C]
        num_channels = input_dim.shape[3]

        out_height = out_size[0]
        out_width = out_size[1]
        grid = _meshgrid(out_height, out_width) # (2, WxH)
        grid = grid.unsqueeze(0)
        grid = torch.reshape(grid, [-1])
        grid = grid.repeat(num_batch)
        grid = torch.reshape(grid, [num_batch, 2, -1]) # (B,2,WxH)

        ## here we do the polar/log-polar transform
        W = torch.tensor(input_dim.shape[1], dtype = torch.double)
        # W = input_dim.shape[1].float()
        maxR = W*radius_factor

        # if radius is from 1 to W/2; log R is from 0 to log(W/2)
        # we map the -1 to +1 grid to log R
        # then remap to 0 to 1
        EXCESS_CONST = 1.1

        logbase = torch.exp(torch.log(W*EXCESS_CONST/2) / W) #10. ** (torch.log10(maxR) / W)
        #torch.exp(torch.log(W*EXCESS_CONST/2) / W) #
        # get radius in pix
        if log:
            # min=1, max=maxR
            r_s = torch.pow(logbase, grid[:, 0, :])
        else:
            # min=1, max=maxR
            r_s = 1 + (grid[:, 0, :] + 1)/2*(maxR-1)

        # y is from -1 to 1; theta is from 0 to 2pi
        theta = np.linspace(0., np.pi, input_dim.shape[1], endpoint=False) * -1.0
        t_s = torch.from_numpy(theta).unsqueeze(1) * torch.ones([1, out_width])
        t_s = torch.reshape(t_s, (1, -1))

        # use + theta[:, 0] to deal with origin
        x_s = r_s*torch.cos(t_s) + (W /2)
        y_s = r_s*torch.sin(t_s) + (W /2)

        x_s_flat = torch.reshape(x_s, [-1])
        y_s_flat = torch.reshape(y_s, [-1])
        
        input_transformed = _interpolate(input_dim, x_s_flat, y_s_flat, out_size)
        output = torch.reshape(input_transformed, [num_batch, out_height, out_width, num_channels]).to(device)
        return output, logbase

    output, logbase = _transform(U, out_size)
    return [output, logbase]

#### Debug
# def meshgrid(height, width):
#     x_t = torch.ones([height, 1]) * torch.linspace(0.0, 1.0 * width-1, width).unsqueeze(1).permute(1, 0)
#     y_t = torch.linspace(0.0, 1.0, height).unsqueeze(1) * torch.ones([1, width])

#     x_t_flat = torch.reshape(x_t, (1, -1))
#     y_t_flat = torch.reshape(y_t, (1, -1))
#     grid = torch.cat((x_t_flat, y_t_flat), 0)

#     return grid

# def transform(input_dim, out_size):
#     radius_factor = torch.sqrt(torch.tensor(2.))/2.
#     log = True
#     num_batch = input_dim.shape[0]  # input [B,W,H,C]
#     num_channels = input_dim.shape[3]
#     # theta = torch.reshape(theta, (-1, 2))
#     # theta = theta.float()

#     out_height = out_size[0]
#     out_width = out_size[1]
#     grid = meshgrid(out_height, out_width) # (2, WxH)
#     grid = grid.unsqueeze(0)
#     grid = torch.reshape(grid, [-1])
#     grid = grid.repeat(num_batch)
#     grid = torch.reshape(grid, [num_batch, 2, -1]) # (B,2,WxH)

#     ## here we do the polar/log-polar transform
#     W = torch.tensor(input_dim.shape[1], dtype = torch.double)
#     # W = input_dim.shape[1].float()
#     maxR = W*radius_factor

#     # if radius is from 1 to W/2; log R is from 0 to log(W/2)
#     # we map the -1 to +1 grid to log R
#     # then remap to 0 to 1

#     EXCESS_CONST = 1.1

#     logbase = torch.exp(torch.log(W*EXCESS_CONST/2) / W) #10. ** (torch.log10(maxR) / W)

#     # get radius in pix
#     if log:
#         # min=1, max=maxR
#         r_s = torch.pow(logbase, grid[:, 0, :])-1
#     else:
#         # min=1, max=maxR
#         r_s = 1 + (grid[:, 0, :] + 1)/2*(maxR-1)
#     # convert it to [0, 2maxR/W]
#     # r_s = (r_s - 1) / (maxR - 1) * 2 * maxR / W
#     # y is from -1 to 1; theta is from 0 to 2pi
#     theta = np.linspace(0, np.pi, W, endpoint=False) * -1.0
#     t_s = torch.from_numpy(theta).unsqueeze(1) * torch.ones([1, out_width])
#     t_s = torch.reshape(t_s, (1, -1))

#     # use + theta[:, 0] to deal with origin
#     x_s = r_s*torch.cos(t_s) + (W /2)#+ theta[:, 0, np.newaxis]  # x
#     y_s = r_s*torch.sin(t_s) + (W /2) #+ theta[:, 1, np.newaxis]

#     x_s_flat = torch.reshape(x_s, [-1])
#     y_s_flat = torch.reshape(y_s, [-1])
    
#     input_transformed = interpolate(input_dim, x_s_flat, y_s_flat, out_size)
#     output = torch.reshape(input_transformed, [num_batch, out_height, out_width, num_channels])
#     return output

# def repeat(x, n_repeats):
#     rep = torch.ones(n_repeats)
#     rep.unsqueeze(0)
#     x = torch.reshape(x, (-1, 1))
#     x = x * rep
#     return torch.reshape(x, [-1])

# def interpolate(im, x, y, out_size): # im [B,H,W,C]
#     # constants
#     num_batch = im.shape[0]
#     height = im.shape[1]
#     width = im.shape[2]
#     channels = im.shape[3]
#     height_f = torch.DoubleTensor(height)
#     width_f = torch.DoubleTensor(width)
    
#     x = x.double()
#     y = y.double()
#     out_height = out_size[0]
#     out_width = out_size[1]
#     zero = torch.zeros([])
#     max_y = im.shape[1] - 1
#     max_x = im.shape[2] - 1

#     # do sampling
#     x0 = torch.floor(x).long()
#     x1 = x0 + 1
#     y0 = torch.floor(y).long()
#     y1 = y0 + 1
#     x0 = torch.clamp(x0, zero, max_x)
#     x1 = torch.clamp(x1, zero, max_x)
#     y0 = torch.clamp(y0, zero, max_y)
#     y1 = torch.clamp(y1, zero, max_y)
#     dim2 = width
#     dim1 = width*height
#     base = repeat(torch.range(0, num_batch-1, dtype=int)*dim1, out_height*out_width)
#     base = base.long()

#     base_y0 = base + y0*dim2
#     base_y1 = base + y1*dim2
#     idx_a = base_y0 + x0
#     idx_b = base_y1 + x0
#     idx_c = base_y0 + x1
#     idx_d = base_y1 + x1

#     # use indices to lookup pixels in the flat image and restore
#     # channels dim
#     im_flat = torch.reshape(im, [-1, channels])
#     im_flat = im_flat.clone().double()

#     Ia = im_flat.gather(0, idx_a.unsqueeze(1))
#     # Ia = im_flat[idx_a-1].to(device)
#     Ib = im_flat.gather(0, idx_b.unsqueeze(1))
#     # Ib = im_flat[idx_b-1].to(device)
#     Ic = im_flat.gather(0, idx_c.unsqueeze(1))
#     # Ic = im_flat[idx_c-1].to(device)
#     Id = im_flat.gather(0, idx_d.unsqueeze(1))
#     # Id = im_flat[idx_d-1].to(device)

#     # and finally calculate interpolated values
#     x0_f = x0.double()
#     x1_f = x1.double()
#     y0_f = y0.double()
#     y1_f = y1.double()
#     wa = ((x1_f-x) * (y1_f-y)).unsqueeze(1)
#     wb = ((x1_f-x) * (y-y0_f)).unsqueeze(1)
#     wc = ((x-x0_f) * (y1_f-y)).unsqueeze(1)
#     wd = ((x-x0_f) * (y-y0_f)).unsqueeze(1)
#     # output = torch.add([wa*Ia, wb*Ib, wc*Ic, wd*Id])
#     output = wa*Ia + wb*Ib + wc*Ic + wd*Id
#     return output

# def imshow(tensor, title=None):
#     unloader = transforms.ToPILImage()
#     image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
#     image = image.squeeze(0)  # remove the fake batch dimension
#     image = unloader(image)
#     plt.imshow(image)
#     plt.show()

# trans = transforms.Compose([
#         transforms.ToTensor(),
#         # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
#     ])

# input = cv2.imread("./1.jpg", 0)
# # # input = polar(input)
# # # cv2.namedWindow("Image")
# # # cv2.imshow("Image",input)
# # # cv2.waitKey (0)

# input = trans(input)
# # imshow(input)
# input = input.unsqueeze(0)
# input = input.permute(0,2,3,1)

# # print(np.shape(input))
# # input = torch.randn((1,128,128,1), dtype=torch.double, requires_grad=True)
# # input = torch.ones(10,128,128,3).requires_grad

# # test = gradcheck(transform, input, eps=1e-6, atol=1e-4)
# # print("Are the gradients correct: ", test)

# output = transform(input, [677,677])

# output_show = output.permute(0,3,1,2)
# output_show = output_show.squeeze(0)
# # print(output_show.shape)
# imshow(output_show.float())
# # output_show = output_show.squeeze(0)

# # print(output_show)
