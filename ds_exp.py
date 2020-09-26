import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
from skimage.metrics import structural_similarity as ssim

model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
#device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
device = torch.device('cpu')

test_img_folder = 'images/4k/*'

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

ds_lr_ssim = []
out_hr_ssim = []
print('Model path {:s}. \nTesting...'.format(model_path))

idx = 0
for path in glob.glob(test_img_folder):
    idx += 1
    base = osp.splitext(osp.basename(path))[0]
    print(idx, base)
    # read images
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    base_img = img.copy()
    ds_factor = 4
    img = cv2.resize(img, (int(img.shape[1]/ds_factor), int(img.shape[0]/ds_factor)), interpolation = cv2.INTER_AREA)
    cv2.imwrite('images/4kDS/{:s}_rlt.png'.format(base), img)

    #LR_cp = cv2.imread('images/LR/{:s}x4.png'.format(base.split('_')[0]), cv2.IMREAD_COLOR)
    #print(ssim(LR_cp, img, multichannel = True))
    #ds_lr_ssim.append(ssim(LR_cp, img, multichannel = True))

    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    #print(img.shape)
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).cpu().data.squeeze().float().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    cv2.imwrite('images/results_4k/{:s}.png'.format(base), output)
    print(ssim(base_img, output, multichannel = True))
    out_hr_ssim.append(ssim(base_img, output, multichannel = True))

print("finished.  avg stats:")
#print("ds-lr ssim:", np.mean(ds_lr_ssim))
print("out-hr ssim:", np.mean(out_hr_ssim))
