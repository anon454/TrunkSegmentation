import os
import numpy as np
import torch
import pickle
from PIL import Image
from utils.misc import AverageMeter, evaluate_incremental, freeze_bn
from utils.segmentor import Segmentor
import torchvision.transforms.functional as F
import cv2

class Validator():
    def __init__(self, data_loader, n_classes=19,
                 save_snapshot=False, extra_name_str=''):
        self.data_loader = data_loader
        self.n_classes = n_classes
        self.save_snapshot = save_snapshot
        self.extra_name_str = extra_name_str

    def run(self, net, optimizer, best_record, curr_iter,
            res_dir, f_handle, iter_max, writer=None):
        # the following code is written assuming that batch size is 1
        net.eval()
        segmentor = Segmentor( net, self.n_classes, colorize_fcn=None,
                n_slices_per_pass=10)

        confmat = np.zeros((self.n_classes, self.n_classes))
        for vi, data in enumerate(self.data_loader):
            if vi==iter_max: # TODO: I do this to save time but once everything
                #works, get rid of this break
                break

            img_slices, gt, slices_info = data
            gt = gt.squeeze_(0).numpy()
            prediction_tmp = segmentor.run_on_slices(
                img_slices.squeeze_(0), slices_info.squeeze_(0))

            if prediction_tmp.shape != gt.shape:
                prediction_tmp = prediction_tmp.astype(np.uint8)
                print(gt.shape)
                prediction_tmp = cv2.resize(prediction_tmp, gt.shape, 
                        cv2.INTER_NEAREST)

            acc, acc_cls, mean_iu, fwavacc, confmat = evaluate_incremental(
                confmat, prediction_tmp, gt, self.n_classes)

            str2write = 'validating: %d / %d' % (vi + 1, iter_max)
            #str2write = 'validating: %d / %d' % (vi + 1, len(self.data_loader))
            print(str2write)
            f_handle.write(str2write + "\n")

        # Store confusion matrix
        confmatdir = '%s/val/confmat'%res_dir
        os.makedirs(confmatdir, exist_ok=True)
        with open(os.path.join(confmatdir, self.extra_name_str + str(curr_iter) + '_confmat.pkl'), 'wb') as confmat_file:
            pickle.dump(confmat, confmat_file)

        
        if self.save_snapshot:
            # save state
            snapshot_name = 'iter_%d_acc_%.5f_acc-cls_%.5f_mean-iu_%.5f_fwavacc_%.5f_lr_%.10f' % (
                curr_iter, acc, acc_cls, mean_iu, fwavacc, optimizer.param_groups[1]['lr'])
            torch.save(net.state_dict(), '%s/snap/%s.pth'%(res_dir, snapshot_name))
            torch.save( optimizer.state_dict(), '%s/snap/opt_%s.pth'%(save_dir,snapshot_name))

            # save perf of this state if it is the new best state
            if best_record['mean_iu'] < mean_iu:
                best_record['iter'] = curr_iter
                best_record['acc'] = acc
                best_record['acc_cls'] = acc_cls
                best_record['mean_iu'] = mean_iu
                best_record['fwavacc'] = fwavacc
                best_record['snapshot'] = snapshot_name
                with open('%s/val/bestval.txt'%res_dir, 'w') as f:
                    f.write( str(best_record) + '\n\n')

            str2write = 'best record: iter: %d\tacc: %.5f\tacc_cls: %.5f\tmean_iu: %.5f\tfwavacc: %.5f]' % (
                    best_record['iter'], best_record['acc'], best_record['acc_cls'], 
                    best_record['mean_iu'], best_record['fwavacc'])
            print(str2write)
            f_handle.write(str2write + "\n")

        str2write = 'Current    : iter: %d\tacc: %.5f\tacc_cls: %.5f\tmean_iu %.5f\tfwavacc %.5f' % (
                curr_iter, acc, acc_cls, mean_iu, fwavacc)
        print(str2write)
        f_handle.write(str2write + "\n")

        if writer is not None:
            writer.add_scalar('acc', acc, curr_iter)
            writer.add_scalar('acc_cls', acc_cls, curr_iter)
            writer.add_scalar('mean_iu', mean_iu, curr_iter)
            writer.add_scalar('fwavacc', fwavacc, curr_iter)

        net.train()
        #if 'freeze_bn' not in args or args.freeze_bn:
        freeze_bn(net)

        return mean_iu
