import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
from skimage.metrics import structural_similarity as ssim

model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
# device = torch.device('cpu')

test_img_folder = 'images/HR/*'

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

print('Model path {:s}. \nTesting...'.format(model_path))

idx = 0
for ds_factor in [2, 4, 8, 16]:
    print("starting ds factor", ds_factor)
    out_hr_ssim = []
    for path in glob.glob(test_img_folder):

        idx += 1
        if idx > 5:
            break
        base = osp.splitext(osp.basename(path))[0]
        #print(idx, base)
        # read images
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        base_img = img.copy()
        img = cv2.resize(img, (int(img.shape[1]/ds_factor), int(img.shape[0]/ds_factor)), interpolation = cv2.INTER_AREA)
        cv2.imwrite('images/DSfact/{:s}_{:s}.png'.format(base, str(ds_factor)), img)

        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        #print(img.shape)
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(device)

        with torch.no_grad():
            tmp = model(img_LR)
            output = tmp.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            del tmp
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
        cv2.imwrite('images/results_dsfact/{:s}_{:s}.png'.format(base, str(ds_factor)), output)
        #print(ssim(base_img, output, multichannel = True))
        print(base)
        out_hr_ssim.append(ssim(base_img, output, multichannel = True))
        del img_LR
        torch.cuda.empty_cache()
        print("done round")

    print("finished.  avg stats:")
    print("out-hr ssim:", np.mean(out_hr_ssim))
