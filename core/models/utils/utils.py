import numpy as np
import torch
import earthnet as en

def mean_cube(cube, mask_channel=1):
    # cube is the input cube (note that the time is always the last coordinate)
    # cannels is the list of channels we compute the avarage on
    # mask_channel whether we include the data quality mask (if we include the mask channel it should be the last one)
    '''
    cube dimensions are:
        b, c, w, h, t
    '''
    channels = cube.shape[1]
    # mask which data is cloudy and shouldn't be used for averaging
    mask = torch.repeat_interleave(1 - cube[:, mask_channel:mask_channel + 1:, :, :, :], channels - 1, axis=1)

    masked_cube = mask * cube[:, :-1, :, :, :] 
    avg_cube = torch.sum(masked_cube, dim=-1) / torch.sum(mask, dim = -1)
    return torch.nan_to_num(avg_cube, nan = 0)

def last_cube(cube, mask_channel=4):
    # note that cube can either be a torch tensor or a numpy array
    new_cube = mean_cube(cube[:, 0:4, :, :, :])

    # for each pixel, find the last good quality data point
    # if no data point has good quality return the mean
    for c in range(cube.shape[0]):
        for i in range(cube.shape[2]):
            for j in range(cube.shape[3]):
                for k in reversed(range(cube.shape[mask_channel])):
                    if cube[c, mask_channel, i, j, k] == 0:
                        new_cube[c, :4, i, j] = cube[c, :4, i, j, k]
                        break
    return new_cube

def last_frame(cube, mask_channel=1):
    # note that by default the last channel will be the mask
    T = cube.shape[-1]
    # 1 = good quality, 0 = bad quality (in the flipped version)
    mask = 1 - cube[:, mask_channel:mask_channel + 1, :, :, T - 1]
    missing = cube[:, mask_channel:mask_channel + 1, :, :, T - 1]  # 1 = is missing, 0 = is already assigned
    new_cube = cube[:, :-1, :, :, T - 1] * mask

    t = T - 1
    while (torch.min(mask) == 0 and t >= 0):
        mask = missing * (1 - cube[:, mask_channel:mask_channel + 1, :, :, t])
        new_cube += cube[:, :-1, :, :, t] * mask
        missing = missing * (1 - mask)
        t -= 1
    return new_cube

def last_season(cube, mask_channel=1):
    # note that by default the last channel will be the mask
    T = cube.shape[-1]
    # 1 = good quality, 0 = bad quality (in the flipped version)
    mask = 1 - cube[:, mask_channel:mask_channel + 1, :, :, T - 4]
    missing = cube[:, mask_channel:mask_channel + 1, :, :, T - 4]  # 1 = is missing, 0 = is already assigned
    new_cube = cube[:, :-1, :, :, T - 4] * mask

    t = T - 4
    while (torch.min(mask) == 0 and t >= 0):
        mask = missing * (1 - cube[:, mask_channel:mask_channel + 1, :, :, t])
        new_cube += cube[:, :-1, :, :, t] * mask
        missing = missing * (1 - mask)
        t -= 4
    return new_cube
    
def zeros(cube, mask_channel = 1):
    return cube[:, :-1, :, :, 0]*0


