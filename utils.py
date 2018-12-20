import sys
import os
import numpy as np
import torch
import path_manager as pm
import SimpleITK as sitk
import pydicom as dicom
import json

def getFreeId():
    import pynvml 

    pynvml.nvmlInit()
    def getFreeRatio(id):
        handle = pynvml.nvmlDeviceGetHandleByIndex(id)
        use = pynvml.nvmlDeviceGetUtilizationRates(handle)
        ratio = 0.5*(float(use.gpu+float(use.memory)))
        return ratio

    deviceCount = pynvml.nvmlDeviceGetCount()
    available = []
    for i in range(deviceCount):
        if getFreeRatio(i)<70:
            available.append(i)
    gpus = ''
    for g in available:
        gpus = gpus+str(g)+','
    gpus = gpus[:-1]
    return gpus

def setgpu(gpuinput):
    freeids = getFreeId()
    if gpuinput=='all':
        gpus = freeids
    else:
        gpus = gpuinput
        if any([g not in freeids for g in gpus.split(',')]):
            raise ValueError('gpu'+g+'is being used')
    print('using gpu '+gpus)
    os.environ['CUDA_VISIBLE_DEVICES']=gpus
    return len(gpus.split(','))

class Logger(object):
    def __init__(self,logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    


def split4(data,  max_stride, margin):
    splits = []
    data = torch.Tensor.numpy(data)
    _,c, z, h, w = data.shape

    w_width = np.ceil(float(w / 2 + margin)/max_stride).astype('int')*max_stride
    h_width = np.ceil(float(h / 2 + margin)/max_stride).astype('int')*max_stride
    pad = int(np.ceil(float(z)/max_stride)*max_stride)-z
    leftpad = pad/2
    pad = [[0,0],[0,0],[leftpad,pad-leftpad],[0,0],[0,0]]
    data = np.pad(data,pad,'constant',constant_values=-1)
    data = torch.from_numpy(data)
    splits.append(data[:, :, :, :h_width, :w_width])
    splits.append(data[:, :, :, :h_width, -w_width:])
    splits.append(data[:, :, :, -h_width:, :w_width])
    splits.append(data[:, :, :, -h_width:, -w_width:])
    
    return torch.cat(splits, 0)

def combine4(output, h, w):
    splits = []
    for i in range(len(output)):
        splits.append(output[i])
 
    output = np.zeros((
        splits[0].shape[0],
        h,
        w,
        splits[0].shape[3],
        splits[0].shape[4]), np.float32)

    h0 = output.shape[1] / 2
    h1 = output.shape[1] - h0
    w0 = output.shape[2] / 2
    w1 = output.shape[2] - w0

    splits[0] = splits[0][:, :h0, :w0, :, :]
    output[:, :h0, :w0, :, :] = splits[0]

    splits[1] = splits[1][:, :h0, -w1:, :, :]
    output[:, :h0, -w1:, :, :] = splits[1]

    splits[2] = splits[2][:, -h1:, :w0, :, :]
    output[:, -h1:, :w0, :, :] = splits[2]

    splits[3] = splits[3][:, -h1:, -w1:, :, :]
    output[:, -h1:, -w1:, :, :] = splits[3]

    return output

def split8(data,  max_stride, margin):
    splits = []
    if isinstance(data, np.ndarray):
        c, z, h, w = data.shape
    else:
        _,c, z, h, w = data.size()
    
    z_width = np.ceil(float(z / 2 + margin)/max_stride).astype('int')*max_stride
    w_width = np.ceil(float(w / 2 + margin)/max_stride).astype('int')*max_stride
    h_width = np.ceil(float(h / 2 + margin)/max_stride).astype('int')*max_stride
    for zz in [[0,z_width],[-z_width,None]]:
        for hh in [[0,h_width],[-h_width,None]]:
            for ww in [[0,w_width],[-w_width,None]]:
                if isinstance(data, np.ndarray):
                    splits.append(data[np.newaxis, :, zz[0]:zz[1], hh[0]:hh[1], ww[0]:ww[1]])
                else:
                    splits.append(data[:, :, zz[0]:zz[1], hh[0]:hh[1], ww[0]:ww[1]])

                
    if isinstance(data, np.ndarray):
        return np.concatenate(splits, 0)
    else:
        return torch.cat(splits, 0)

    

def combine8(output, z, h, w):
    splits = []
    for i in range(len(output)):
        splits.append(output[i])
 
    output = np.zeros((
        z,
        h,
        w,
        splits[0].shape[3],
        splits[0].shape[4]), np.float32)

    
    z_width = z / 2
    h_width = h / 2
    w_width = w / 2
    i = 0
    for zz in [[0,z_width],[z_width-z,None]]:
        for hh in [[0,h_width],[h_width-h,None]]:
            for ww in [[0,w_width],[w_width-w,None]]:
                output[zz[0]:zz[1], hh[0]:hh[1], ww[0]:ww[1], :, :] = splits[i][zz[0]:zz[1], hh[0]:hh[1], ww[0]:ww[1], :, :]
                i = i+1
                
    return output


def split16(data,  max_stride, margin):
    splits = []
    _,c, z, h, w = data.size()
    
    z_width = np.ceil(float(z / 4 + margin)/max_stride).astype('int')*max_stride
    z_pos = [z*3/8-z_width/2,
             z*5/8-z_width/2]
    h_width = np.ceil(float(h / 2 + margin)/max_stride).astype('int')*max_stride
    w_width = np.ceil(float(w / 2 + margin)/max_stride).astype('int')*max_stride
    for zz in [[0,z_width],[z_pos[0],z_pos[0]+z_width],[z_pos[1],z_pos[1]+z_width],[-z_width,None]]:
        for hh in [[0,h_width],[-h_width,None]]:
            for ww in [[0,w_width],[-w_width,None]]:
                splits.append(data[:, :, zz[0]:zz[1], hh[0]:hh[1], ww[0]:ww[1]])
    
    return torch.cat(splits, 0)

def combine16(output, z, h, w):
    splits = []
    for i in range(len(output)):
        splits.append(output[i])
 
    output = np.zeros((
        z,
        h,
        w,
        splits[0].shape[3],
        splits[0].shape[4]), np.float32)

    
    z_width = z / 4
    h_width = h / 2
    w_width = w / 2
    splitzstart = splits[0].shape[0]/2-z_width/2
    z_pos = [z*3/8-z_width/2,
             z*5/8-z_width/2]
    i = 0
    for zz,zz2 in zip([[0,z_width],[z_width,z_width*2],[z_width*2,z_width*3],[z_width*3-z,None]],
                      [[0,z_width],[splitzstart,z_width+splitzstart],[splitzstart,z_width+splitzstart],[z_width*3-z,None]]):
        for hh in [[0,h_width],[h_width-h,None]]:
            for ww in [[0,w_width],[w_width-w,None]]:
                output[zz[0]:zz[1], hh[0]:hh[1], ww[0]:ww[1], :, :] = splits[i][zz2[0]:zz2[1], hh[0]:hh[1], ww[0]:ww[1], :, :]
                i = i+1
                
    return output

def split32(data,  max_stride, margin):
    splits = []
    _,c, z, h, w = data.size()
    
    z_width = np.ceil(float(z / 2 + margin)/max_stride).astype('int')*max_stride
    w_width = np.ceil(float(w / 4 + margin)/max_stride).astype('int')*max_stride
    h_width = np.ceil(float(h / 4 + margin)/max_stride).astype('int')*max_stride
    
    w_pos = [w*3/8-w_width/2,
             w*5/8-w_width/2]
    h_pos = [h*3/8-h_width/2,
             h*5/8-h_width/2]

    for zz in [[0,z_width],[-z_width,None]]:
        for hh in [[0,h_width],[h_pos[0],h_pos[0]+h_width],[h_pos[1],h_pos[1]+h_width],[-h_width,None]]:
            for ww in [[0,w_width],[w_pos[0],w_pos[0]+w_width],[w_pos[1],w_pos[1]+w_width],[-w_width,None]]:
                splits.append(data[:, :, zz[0]:zz[1], hh[0]:hh[1], ww[0]:ww[1]])
    
    return torch.cat(splits, 0)

def combine32(splits, z, h, w):
 
    output = np.zeros((
        z,
        h,
        w,
        splits[0].shape[3],
        splits[0].shape[4]), np.float32)

    
    z_width = int(np.ceil(float(z) / 2))
    h_width = int(np.ceil(float(h) / 4))
    w_width = int(np.ceil(float(w) / 4))
    splithstart = splits[0].shape[1]/2-h_width/2
    splitwstart = splits[0].shape[2]/2-w_width/2
    
    i = 0
    for zz in [[0,z_width],[z_width-z,None]]:
        
        for hh,hh2 in zip([[0,h_width],[h_width,h_width*2],[h_width*2,h_width*3],[h_width*3-h,None]],
                          [[0,h_width],[splithstart,h_width+splithstart],[splithstart,h_width+splithstart],[h_width*3-h,None]]):
            
            for ww,ww2 in zip([[0,w_width],[w_width,w_width*2],[w_width*2,w_width*3],[w_width*3-w,None]],
                              [[0,w_width],[splitwstart,w_width+splitwstart],[splitwstart,w_width+splitwstart],[w_width*3-w,None]]):
                
                output[zz[0]:zz[1], hh[0]:hh[1], ww[0]:ww[1], :, :] = splits[i][zz[0]:zz[1], hh2[0]:hh2[1], ww2[0]:ww2[1], :, :]
                i = i+1
                
    return output



def split64(data,  max_stride, margin):
    splits = []
    _,c, z, h, w = data.size()
    
    z_width = np.ceil(float(z / 4 + margin)/max_stride).astype('int')*max_stride
    w_width = np.ceil(float(w / 4 + margin)/max_stride).astype('int')*max_stride
    h_width = np.ceil(float(h / 4 + margin)/max_stride).astype('int')*max_stride
    
    z_pos = [z*3/8-z_width/2,
             z*5/8-z_width/2]
    w_pos = [w*3/8-w_width/2,
             w*5/8-w_width/2]
    h_pos = [h*3/8-h_width/2,
             h*5/8-h_width/2]

    for zz in [[0,z_width],[z_pos[0],z_pos[0]+z_width],[z_pos[1],z_pos[1]+z_width],[-z_width,None]]:
        for hh in [[0,h_width],[h_pos[0],h_pos[0]+h_width],[h_pos[1],h_pos[1]+h_width],[-h_width,None]]:
            for ww in [[0,w_width],[w_pos[0],w_pos[0]+w_width],[w_pos[1],w_pos[1]+w_width],[-w_width,None]]:
                splits.append(data[:, :, zz[0]:zz[1], hh[0]:hh[1], ww[0]:ww[1]])
    
    return torch.cat(splits, 0)

def combine64(output, z, h, w):
    splits = []
    for i in range(len(output)):
        splits.append(output[i])
 
    output = np.zeros((
        z,
        h,
        w,
        splits[0].shape[3],
        splits[0].shape[4]), np.float32)

    
    z_width = int(np.ceil(float(z) / 4))
    h_width = int(np.ceil(float(h) / 4))
    w_width = int(np.ceil(float(w) / 4))
    splitzstart = splits[0].shape[0]/2-z_width/2
    splithstart = splits[0].shape[1]/2-h_width/2
    splitwstart = splits[0].shape[2]/2-w_width/2
    
    i = 0
    for zz,zz2 in zip([[0,z_width],[z_width,z_width*2],[z_width*2,z_width*3],[z_width*3-z,None]],
                          [[0,z_width],[splitzstart,z_width+splitzstart],[splitzstart,z_width+splitzstart],[z_width*3-z,None]]):
        
        for hh,hh2 in zip([[0,h_width],[h_width,h_width*2],[h_width*2,h_width*3],[h_width*3-h,None]],
                          [[0,h_width],[splithstart,h_width+splithstart],[splithstart,h_width+splithstart],[h_width*3-h,None]]):
            
            for ww,ww2 in zip([[0,w_width],[w_width,w_width*2],[w_width*2,w_width*3],[w_width*3-w,None]],
                              [[0,w_width],[splitwstart,w_width+splitwstart],[splitwstart,w_width+splitwstart],[w_width*3-w,None]]):
                
                output[zz[0]:zz[1], hh[0]:hh[1], ww[0]:ww[1], :, :] = splits[i][zz2[0]:zz2[1], hh2[0]:hh2[1], ww2[0]:ww2[1], :, :]
                i = i+1
                
    return output


import numpy as np
import SimpleITK as sitk
import scipy.ndimage as snd
import skimage.morphology as morph

# TODO: Query variables from a CONFIG file
def HU_window(image, hu_min=-100, hu_max=400):
    return np.clip(image, hu_min, hu_max)

def resample(
    image, output_spacing, output_origin, output_direction,
    interpolator=sitk.sitkLinear, default_value=0):
    """
    image : sitk Image
    """
    input_size = image.GetSize()
    input_spacing = image.GetSpacing()
    output_size = [0.0, 0.0, 0.0]

    output_size[0] = int(input_size[0] * input_spacing[0] / output_spacing[0] + .5)
    output_size[1] = int(input_size[1] * input_spacing[1] / output_spacing[1] + .5)
    output_size[2] = int(input_size[2] * input_spacing[2] / output_spacing[2] + .5)

    output_size = tuple(output_size)

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(output_size)
    resampler.SetOutputSpacing(output_spacing)
    resampler.SetOutputOrigin(output_origin)
    resampler.SetOutputDirection(output_direction)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_value)
    image = resampler.Execute(image)
    return image

def resample_back(
    image, output_size, output_spacing, output_origin, output_direction,
    interpolator=sitk.sitkLinear, default_value=0):
    input_size = image.GetSize()
    input_spacing = image.GetSpacing()
   
    output_size = tuple(output_size)

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(output_size)
    resampler.SetOutputSpacing(output_spacing)
    resampler.SetOutputOrigin(output_origin)
    resampler.SetOutputDirection(output_direction)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_value)
    image = resampler.Execute(image)
    return image

def normalize(image):
    image = np.float32(image)
    return (image - image.min()) / (image.max() - image.min())

def twoSplit(x):
    return x//2, x - x//2

def calc_pad(img_shape, vol_size):
    new_shape = [0,0,0]

    if img_shape[0] % vol_size[0] != 0:
        new_shape[0] = (img_shape[0]//vol_size[0] + 1)*vol_size[0]
    else:
        new_shape[0] = img_shape[0]

    if img_shape[1] % vol_size[1] != 0:
        new_shape[1] = (img_shape[1]//vol_size[1] + 1)*vol_size[1]
    else:
        new_shape[1] = img_shape[1]

    if img_shape[2] % vol_size[2] != 0:
        new_shape[2] = (img_shape[2]//vol_size[2] + 1)*vol_size[2]
    else:
        new_shape[2] = img_shape[2]

    diff_x, diff_y, diff_z = (new_shape[0] - img_shape[0]), \
                             (new_shape[1] - img_shape[1]), \
                             (new_shape[2] - img_shape[2])

    # before_x, before_y, before_z = diff_x//2, diff_y//2, diff_z//2
    before_x, before_y, before_z = 0, 0, 0
    after_x, after_y, after_z = diff_x - before_x, \
                                diff_y - before_y, \
                                diff_z - before_z
    pad_width  = ((before_x, after_x),
                  (before_y, after_y),
                  (before_z, after_z))
    return pad_width

def crop_pad_width(image, pad_width):
    image_shape = image.shape
    before_x, after_x = pad_width[0]
    before_y, after_y = pad_width[1]
    before_z, after_z = pad_width[2]

    if image.ndim == 3:
        return image[before_x: image_shape[0] -after_x, 
                     before_y: image_shape[1] -after_y, 
                     before_z: image_shape[2] -after_z]
    elif image.ndim == 4:
        return image[ before_x: image_shape[0] -after_x, 
                     before_y: image_shape[1] -after_y, 
                     before_z: image_shape[2] -after_z, :]
    
def pad(image, pad_size, mode='constant'):
    return np.pad(image, pad_size, mode=mode)

def pad_after_prediction(image, to_size, patch_size, out_size):
    """
    to_size: [x, y, z] extend image to the match `to_size`
    """
    before = [0, 0, 0]
    after = [0, 0, 0]

    for i in range(3):
        before[i] = (patch_size[i] - out_size[i]) // 2
        after[i] = to_size[i] - image.shape[i] - before[i]

    pad_width = list(zip(before, after))
    return pad(image, pad_width)

def crop_pad_width(image, pad_width):
    # print(pad_width)
    image_shape = image.shape
    before_x, after_x = pad_width[0]
    before_y, after_y = pad_width[1]
    before_z, after_z = pad_width[2]

    if image.ndim == 3:
        return image[before_x: image_shape[0] -after_x, 
                     before_y: image_shape[1] -after_y, 
                     before_z: image_shape[2] -after_z]
    elif image.ndim == 4:
        return image[ before_x: image_shape[0] -after_x, 
                     before_y: image_shape[1] -after_y, 
                     before_z: image_shape[2] -after_z, :]

def retain_largest(img):
    """
    img : 3D numpy array
    Retains the largest connected component in the binary image
    """
    mask = img > 0
    c, n = snd.label(mask)

    sizes = snd.sum(mask, c, range(n + 1))
    mask_size = sizes < max(sizes)
    remove_voxels = mask_size[c]
    c[remove_voxels] = 0
    c[np.where(c != 0)] = 1
    return c

################## KC's addition ###################
def setVisibleDevices(gpu_ids):
    assert type(gpu_ids) == str
    print("GPU ids : " + str(gpu_ids))
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    gpu_ids = list(range(0,len(gpu_ids.split(','))))
    return gpu_ids

def load_scan(path):

    slices = [
        dicom.read_file(os.path.join(path, f), force=True) for f in os.listdir(path)
    ]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    if slices[0].ImagePositionPatient[2] == slices[1].ImagePositionPatient[2]:
        sec_num = 2
        while (
            slices[0].ImagePositionPatient[2] == slices[sec_num].ImagePositionPatient[2]
        ):
            sec_num = sec_num + 1
        slice_num = int(len(slices) / sec_num)
        slices.sort(key=lambda x: float(x.InstanceNumber))
        slices = slices[0:slice_num]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(
            slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2]
        )
    except Exception as e:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def get_pixels(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    spacing = [[slices[0].SliceThickness], slices[0].PixelSpacing]
    spacing = np.concatenate(spacing).astype(float)
    spacing = tuple(list(spacing))

    return (
        np.array(image, dtype=np.int16),
        spacing,
    )


def load_dicom_series(series_path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(series_path)
    reader.SetFileNames(dicom_names)
    return reader.Execute()


def load_dicom_image(path_to_image):
    assert(os.path.exists(path_to_image)), "File path " + path_to_image +  " does not exist"
    img = None
    if os.path.isfile(path_to_image):
        img = sitk.ReadImage(path_to_image)
    elif os.path.isdir(path_to_image):
        # img = load_dicom_series(path_to_image)
        try:
            img = load_dicom_series(path_to_image)
        except Exception as e:
            print(str(e))
            print("Failed with SimpleITK, trying with pydicom")
            slices = load_scan(path_to_image)
            pixel_array, spacing = get_pixels(slices)
            img = sitk.GetImageFromArray(pixel_array)
            img.SetSpacing(list(reversed(spacing)))

        spacing = img.GetSpacing()

        if spacing[2] >= 10.0:
            print("Slice thickness is too high, trying with pydicom")

            try:
                slices = load_scan(path_to_image)
                pixel_array, spacing = get_pixels(slices)
                img = sitk.GetImageFromArray(pixel_array)
                img.SetSpacing(list(reversed(spacing)))
                spacing = img.GetSpacing()
            except Exception as e:
                print("Failed with pydicom")
        if spacing[2] > 10.0:
            return None
        # assert (spacing[2] <= 10.0), "Slice-thickness is > 5mm and is not suitable for processing. Please check image acquisition"
    else:
        raise()
        print('sitk_funcs.load_dicom_image:  Unsupported image type')
    return img

def fovia_to_voxel(centroid, size, spacing):
    centroid, size, spacing = np.array(centroid), np.array(size), np.array(spacing)
    volume_box = (size / 2) * spacing
    centroid = ((centroid + volume_box) / spacing).astype(int)
    return centroid

def log_progress(sequence, every=None, size=None, name='Items'):
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )

################# - Curation utils - ####################
def getLocation(num):
    if np.isnan(num):
        return 'NA'
    elif num == 1:
        return 'Right Upper Lobe'
    elif num == 2:
        return 'Right Middle Lobe'
    elif num == 3:
        return 'Right Lower Lobe'
    elif num == 4:
        return 'Left Upper Lobe'
    elif num == 5:
        return 'Lingula'
    elif num == 6:
        return 'Left Lower Lobe'
    elif num == 8:
        return 'Other'

def getTexture(num):
    if np.isnan(num):
        return 'NA'
    elif num == 1:
        return 'Soft Tissue'
    elif num == 2:
        return 'Ground Glass'
    elif num == 3:
        return 'Mixed'
    elif num == 4:
        return 'Fluid or Water'
    elif num == 6:
        return 'Fat'
    elif num == 7:
        return 'Other'
    elif num == 9:
        return 'Unable to determine'

def getFolderFromStudyYear(path, study_year):
    folds = os.listdir(path)
    if study_year == 0:
        for each in folds:
            if '1999' in each:
                return each
        else:
            return None
    elif study_year == 1:
        for each in folds:
            if '2000' in each:
                return each
        else:
            return None
    elif study_year == 2:
        for each in folds:
            if '2001' in each:
                return each
        else:
            return None
    else:
        return None

def assertStudYear(each, study_folder, num):
    if each == 0:
        assert '1999' in study_folder, 'no 1999 in '+str(num)+'/' + str(each)
    elif each == 1:
        assert '2000' in study_folder, 'no 2000 in '+str(num)+'/' + str(each)
    elif each == 2:
        assert '2001' in study_folder, 'no 2001 in '+str(num)+'/' + str(each)
    else:
        print(pid,each)
        print('study year number out of range')
        raise 100


def get_series_dict(folder):
    import pydicom as dicom
    from collections import OrderedDict
    import SimpleITK as sitk
    
    Lung = "Lung"
    Mediastinum = "Mediastinum"
    kernels = {
        "STANDARD": Mediastinum,
        "Bone": Lung,
        "B": Mediastinum,
        "D": Lung,
        "B30f": Mediastinum,
        "B50f": Lung,
        "FC51": Mediastinum,
        "C": Mediastinum,
        "LUNG": Lung,
        "FC01": Mediastinum,
        "FC10": Mediastinum,
        "B80f": Lung,
        "B60f": Lung,
        "FC30": Lung,
        "FC70f": Lung,
        "FC02": Mediastinum,
        "B20f": Mediastinum,
        "A": Mediastinum,
        "FC82": Lung,
        "FC50": Lung,
        "B30f": Mediastinum,
        "B50f": Lung,
        "B60f": Lung,
        "LUNG": Lung,
        "FC01": Mediastinum,
        "FC10": Mediastinum,
        "Standard": Mediastinum,
        "FC51": Mediastinum,
        "FC30": Lung,
        "BONE": Lung,
        "B20f": Mediastinum,
        "C": Mediastinum,
        "D": Lung,
        "FC82": Lung,
        "FC02": Mediastinum,
        "EC": Mediastinum,
        "FC50": Lung,
        "B": Mediastinum,
        "B30s": Mediastinum,
        "B50s": Lung,
        "B45f": Lung,
        "B70f": Lung,
        "B35f": Mediastinum,
        "B31f": Mediastinum,
        "B60s": Lung,
    }

    kernels = {k.lower(): kernels[k] for k in kernels.keys()}

    series_list = {}

    for subdir, dir, files in os.walk(folder):
        for f in files:
            if ".dcm" in f:
                if subdir not in series_list.keys():
                    series_list[subdir] = []
                series_list[subdir].append(f)

    # keep series only if there are at least 100 slices
    unique_series = {
        k: series_list[k] for k in series_list.keys() if len(series_list[k]) > 100
    }
    series_list = unique_series
    series_kernels = []

    # find convolution kernel used for CT reconstruction
    for k in series_list.keys():
        dcms = sorted([os.path.join(k, f) for f in os.listdir(k)])
        d = dicom.read_file(dcms[0])
        k = kernels[d.ConvolutionKernel.lower()]
        series_kernels.append(k)

    dates = {}

    # keep series with lung kernel filtered if there are multiple CTs with same source CT
    for series, kernel in zip(series_list.keys(), series_kernels):
        date = series.split("-NLST")[0].split("/")[-1]
        if date not in dates.keys():
            dates[date] = series
        else:
            if kernel == "Lung":
                dates[date] = series

    sorted_dates = sorted(dates.keys(), key=lambda x: int(x.split("-")[-1]))
    final_dates = OrderedDict()
    for k in sorted_dates:
        final_dates[k] = dates[k]

    return final_dates
        
def getSeriesFolder(patient_folder, study_folder):
    series_dict = get_series_dict(patient_folder)
    year = study_folder.split('-NLST')[0]
    try:
        return series_dict[year]
    except KeyError:
        return None

def HUWindow(arr, mn =-1000, mx=400):
    return np.clip(arr,mn,mx)

def preProcessImage(image,hu_range):
    image = HUWindow(image,hu_range[0], hu_range[1])
    return image

def getLabelAndPatientFolder(dt, pid, images_root):
    label_desc = dt[pid]
    if label_desc == 'Follow-up Collected - Confirmed Not Lung Cancer':
        label = 0
        fold_root = 'NLST/'
    elif label_desc == 'Follow-up collected - Confirmed Lung Cancer':
        label = 1
        fold_root = 'NLST_Cancer/'
    else:
        label = None
        fold_root = ''
    
    patient_folder = pm.join(images_root,fold_root,str(pid))
    
    return label, patient_folder

def getAttributesForValidPid(dt, pid, nlst):
    valid = False
    if pid in nlst:
        valid = True
        slc_locs = []
        locations = []
        textures = []
        diams = []
        l_diams = []
        study_years = []
        for each in dt[pid]['info']:
            slc_locs.append(each[7])
            study_years.append(each[0])
            textures.append(getTexture(each[2]))
            locations.append(getLocation(each[3]))
            diams.append(each[4])
            l_diams.append(each[5])
        return valid, slc_locs, locations, textures, diams, l_diams, study_years
    else:
        return [None]*7
    
def findIfReverse(series_folder):
    lst = os.listdir(series_folder)
    lst.sort()
    r = dicom.read_file(pm.join(series_folder,lst[0]))
    if r.InstanceNumber == 1:
        reverse = False
    else:
        reverse = True
    return reverse
    
def loadAndPreProcessImage(series_folder,hu_range=[-1000,400]):
#     reverse = findIfReverse(series_folder)
    img = load_dicom_image(series_folder)
    if img != None:
        spacing, size = img.GetSpacing(),img.GetSize()
        image = sitk.GetArrayFromImage(img)
        image = preProcessImage(image,hu_range)
        return image, spacing, size
    else:
        return None,None,None

def getCentroidsAndDiameters(label_file_path, spacing, size):
    lbl_data = getLabelData(label_file_path)
    centroids = []
    centroids_z = []
    sideLengths_z = []
    diameters = []
    isNodule = True
    if label_file_path.endswith('.npynew'):
        for d in lbl_data:
            corners = [[d["corners"][i]["x"], d["corners"][i]["y"], d["corners"][i]["z"]] for i in range(len(d["corners"]))]
            centroid = np.array(corners).mean(axis=0)
            centroid = fovia_to_voxel(centroid, size, spacing)
            centroid = list(reversed(centroid))
            centroids.append(centroid)
            centroids_z.append(centroid[0])

            corners = np.array(corners)
            x, y, z = (corners[1] - corners[0]).sum() * spacing[0], (corners[2] - corners[0]).sum() * spacing[1], (corners[4]-corners[0]).sum() * spacing[2]
            sideLengths_z.append(z)
            diameters.append(np.mean((x,y)))
    elif label_file_path.endswith('.npy'):
        if len(list(lbl_data.keys())) != 0:
            for i in lbl_data:
                centroid = lbl_data[i]['centroid']
                side_length = np.array(lbl_data[i]['side_length'])
                centroids.append(centroid)
                centroids_z.append(centroid[0])
                sideLengths_z.append(side_length[0])
                diameters.append(np.mean((side_length[1],side_length[2])))
        else:
            isNodule = False
    return isNodule, centroids, centroids_z, sideLengths_z, diameters

def getLabelFilePath(series_folder, labels_root):
    label_file_path = pm.join(*(series_folder.split('/')[7:])).replace('/','_x_') + '.npynew'
    fold_root = series_folder.split('/')[6]
    label_file_path = pm.join(labels_root,fold_root,'json_path',label_file_path)
    if pm.isfile(label_file_path):
        return label_file_path
    else:
        return label_file_path[:-3]

def getLungViewer2Link(series_folder, label_file_path):
    if label_file_path.endswith('.npynew'):
        print('http://192.168.3.3:5000/lung_viewer2?path='+series_folder+'&json='+label_file_path)
    elif label_file_path.endswith('.npy'):
        print('http://192.168.3.3:5000/lung_viewer?path='+series_folder+'&json='+label_file_path)

def getLabelData(label_file_path):
    if label_file_path.endswith('.npynew'):
        with open(label_file_path) as f:
            data = f.read()
        lbl_data = json.loads(data)
    elif label_file_path.endswith('.npy'):
        lbl_data = np.load(label_file_path).item()
    return lbl_data

def padImage(img,d):
    return np.pad(img,d,mode='constant')

def getAxisWiseImages(image, z, y, x, d):
#     image = padImage(image,d)
    try:
        save_img_z_bef = image[z-1,y-d:y+d,x-d:x+d]
    except Exception as e:
        img = image.copy()
        img = np.pad(img,[[0,0],[d,d],[d,d]])
        save_img_z_bef = img[z-1,y:y+2*d,x:x+2*d]
    
    try:
        save_img_z = image[z,y-d:y+d,x-d:x+d]
    except Exception as e:
        img = image.copy()
        img = np.pad(img,[[0,0],[d,d],[d,d]])
        save_img_z = img[z,y:y+2*d,x:x+2*d]
    
    try:
        save_img_z_af = image[z+1,y-d:y+d,x-d:x+d]
    except Exception as e:
        img = image.copy()
        img = np.pad(img,[[0,0],[d,d],[d,d]])
        save_img_z_af = img[z+1,y:y+2*d,x:x+2*d]
    
    return save_img_z_bef, save_img_z, save_img_z_af 
#     try:
#         save_img_y = image[z-d:z+d,y,x-d:x+d]
#     except Exception as e:
#         img = image.copy()
#         img = np.pad(img,[[d,d],[0,0],[d,d]])
#         save_img_y = img[z:z+2*d,y,x:x+2*d]
    
#     try:
#         save_img_x = image[z-d:z+d,y-d:y+d,x]
#     except Exception as e:
#         img = image.copy()
#         img = np.pad(img,[[d,d],[d,d],[0,0]])
#         save_img_x = img[z:z+2*d,y:y+2*d,x]
    
    
#     return save_img_z, save_img_y, save_img_x
