import os
import shutil
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import CrossEntropyLoss
from torch.nn.functional import binary_cross_entropy
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import numpy as np
from scipy import ndimage as ndi
from tqdm import tqdm

from utils import imgs as img_utils
from torchvision.utils import save_image
import torch.nn.functional as F

from utils.general_utils import list_ave, save_tensor_batch


def show_batch(dl):
    for images, labels in dl:
        print(labels.size())
        print(images.size())
        for l in labels[0]:
            plt.imshow(l)
            plt.show()
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([]);
        ax.set_yticks([])
        ax.imshow(make_grid(images[0][:900], nrow=10, normalize=True).permute(1, 2, 0))
        plt.show()
        break


def save_weights(model, optim, tag, folder, epoch, loss, err, dice, history_loss_t, history_loss_v, history_accuracy_t,
                 history_accuracy_v, history_DSC, history_sens_v, history_spec_v, weights_path):
    weights_fname = 'weights-%d-%d-%d.pth' % (tag, folder, epoch)
    weights_fpath = os.path.join(weights_path, weights_fname)
    torch.save({
        'startEpoch': epoch + 1,
        'loss': loss,
        'error': err,
        'dice': dice,
        'model_state': model.state_dict(),
        'optim_state': optim.state_dict(),
        'history_loss_t': history_loss_t,
        'history_loss_v': history_loss_v,
        'history_accuracy_t': history_accuracy_t,
        'history_accuracy_v': history_accuracy_v,
        'history_DSC': history_DSC,
        'history_sens_v': history_sens_v,
        'history_spec_v': history_spec_v
    }, weights_fpath)
    shutil.copyfile(weights_fpath, os.path.join(weights_path, 'latest.th'))


def load_weights(model, optimizer, fpath):
    print("loading weights '{}'".format(fpath))
    weights = torch.load(fpath)
    startEpoch = weights['startEpoch']
    history_loss_t = weights['history_loss_t']
    history_loss_v = weights['history_loss_v']
    history_accuracy_t = weights['history_accuracy_t']
    history_accuracy_v = weights['history_accuracy_v']
    history_DSC = weights['history_DSC']
    history_sens_v = weights['history_sens_v']
    history_spec_v = weights['history_spec_v']
    model.load_state_dict(weights['model_state'])
    optimizer.load_state_dict(weights['optim_state'])
    print("loaded weights (lastEpoch {}, loss {}, error {}, dice {})"
          .format(startEpoch - 1, weights['loss'], weights['error'], weights['dice']))
    return startEpoch, history_loss_t, history_loss_v, history_accuracy_t, history_accuracy_v, history_DSC, history_sens_v, history_spec_v


def adjust_learning_rate(lr, decay, optimizer, cur_epoch, n_epochs):
    """Sets the learning rate to the initially
        configured `lr` decayed by `decay` every `n_epochs`"""
    new_lr = lr * (decay ** (cur_epoch // n_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()


def error(preds, targets):
    bs, h, w = preds.size()
    n_pixels = bs * h * w
    incorrect = preds.ne(targets).cpu().sum().item()
    err = incorrect / n_pixels
    return round(err, 5)








def train(model, trn_loader, optimizer, criterion, seq_size, sliding_window, loss_type, epoch, epochs, prev_metrics,
          writer, global_step):
    model.train()
    sum_trn_loss = 0
    trn_error = 0
    trn_tp = 0
    trn_fp = 0
    trn_fn = 0
    trn_tn = 0
    seq_window = (seq_size - 1) // 2
    mean_loss = 0
    mean_acc = 0
    mean_dsc = 0

    progress_bar = tqdm(enumerate(trn_loader), total=len(trn_loader))

    for idx, data in progress_bar:
        global_step += 1
        inputs = data[0].cuda()
        targets = data[1][:, 1, :, :].cuda()

        optimizer.zero_grad()
        net_out = model(inputs)

        loss = criterion(net_out, targets.long())
        loss.backward()
        optimizer.step()

        outputs = F.softmax(net_out, dim=1)[:, 1, :, :]
        iter_loss = loss.item()
        sum_trn_loss += iter_loss
        prev_metrics["history_loss_train"].append(iter_loss)

        preds = get_predictions(outputs)

        # err = error(preds, targets)
        # trn_error += err
        # prev_metrics["history_acc_train"].append(1-err)

        tmp_tp, tmp_fp, tmp_fn, tmp_tn = compute_performance(preds, targets)

        prev_metrics["history_DSC_train"].append(dice(tmp_tp, tmp_fp, tmp_fn))

        trn_tp += tmp_tp
        trn_fp += tmp_fp
        trn_fn += tmp_fn
        trn_tn += tmp_tn

        if (idx + 1) % 5 == 0 or idx == len(trn_loader) - 1:
            mean_loss = list_ave(prev_metrics["history_loss_train"][-20:])
            writer.add_scalar("loss_train", mean_loss, global_step)

            # mean_acc = list_ave(prev_metrics["history_acc_train"][-20:])
            # writer.add_scalar("Accuracy_train",mean_acc, global_step)

            mean_dsc = list_ave(prev_metrics["history_DSC_train"][-20:])
            writer.add_scalar("DSC_train", mean_dsc, global_step)

            # save_tensor_batch(outputs.unsqueeze(1), 'results_ms/images', f"0-out{idx}", False)

        cmd_label = f"[{epoch}/{epochs}], Loss:{mean_loss:.5f}, DSC:{mean_dsc:.5f}"
        progress_bar.set_description(cmd_label)

    trn_size = len(trn_loader)
    sum_trn_loss /= trn_size
    trn_error /= trn_size
    trn_dice = dice(trn_tp, trn_fp, trn_fn)
    sens = trn_tp / (trn_tp + trn_fn)
    spec = trn_tn / (trn_tn + trn_fp)
    return sum_trn_loss, trn_error, trn_dice, sens, spec, global_step


def test(model, test_loader, criterion, seq_size, sliding_window, loss_type, writer, epoch, images_dir):
    model.eval()
    test_loss = 0
    test_error = 0
    test_tp = 0
    test_fp = 0
    test_fn = 0
    test_tn = 0
    for idx, (inputs, targets) in enumerate(test_loader):
        with torch.no_grad():
            inputs = inputs.cuda()
            targets = targets.cuda()
            targets = targets[:, 1, :, :]
            net_out = model(inputs)
            test_loss += criterion(net_out, targets.long()).item()

            outputs = F.softmax(net_out, dim=1)[:, 1, :, :]

            preds = get_predictions(outputs)
            tmp_tp, tmp_fp, tmp_fn, tmp_tn = compute_performance(preds, targets)
            test_tp += tmp_tp
            test_fp += tmp_fp
            test_fn += tmp_fn
            test_tn += tmp_tn

            if (idx == 0):
                save_tensor_batch(preds.unsqueeze(1).float(), images_dir, f"predict{epoch:03d}", False)

    test_loss /= len(test_loader)

    writer.add_scalar("Loss_val", test_loss, epoch)

    test_dice = dice(test_tp, test_fp, test_fn)
    writer.add_scalar("DSC_val", test_dice, epoch)

    sens = test_tp / (test_tp + test_fn)
    spec = test_tn / (test_tn + test_fp)
    ppv = test_tp / (test_tp + test_fp) if (test_tp + test_fp) != 0 else 0
    npv = test_tn / (test_tn + test_fn)
    return test_loss, test_error, test_dice, sens, spec, ppv, npv


# save slice + ground truth + prediction + FP + FN
def compute_output(model, test_loader, output_path, seq_size, sliding_window):
    if sliding_window:
        curr_seq_size = 1
        image_idx = seq_size // 2
    else:
        curr_seq_size = seq_size
        image_idx = 0
    test_tp = 0
    test_fp = 0
    test_fn = 0
    test_tn = 0
    dice_vector = []
    for inputs, targets in test_loader:
        with torch.no_grad():
            inputs = inputs.cuda()
            targets = targets.cuda()
            targets = targets.view(seq_size, targets.size(2), targets.size(3))
            outputs = model(inputs)[0]
            inputs = inputs.view(seq_size, inputs.size(2), inputs.size(3), inputs.size(4))

            if sliding_window:
                seq_window = (seq_size - 1) // 2
                indices = range(seq_window, outputs.size(0), seq_size)
                inputs = inputs[indices, :, :, :]
                outputs = outputs[indices, :, :, :]
                targets = targets[indices, :, :]
            pred = get_predictions(outputs)
            imgs_to_save = []
            for j in range(curr_seq_size):
                np_pred = pred[j]
                ejtema = np.zeros((160, 160))
                notditectedlesion = np.zeros((160, 160))
                wronglesiondetected = np.zeros((160, 160))
                b = targets[j]
                for i in range(b.shape[0] - 1):
                    for k in range(b.shape[1] - 1):

                        if b[i, k] == 1 or np_pred[i, k] == 1:
                            ejtema[i, k] = 1
                            if b[i, k] == 1 and np_pred[i, k] == 0:
                                notditectedlesion[i, k] = 1
                            if b[i, k] == 0 and np_pred[i, k] == 1:
                                wronglesiondetected[i, k] = 1

                false_negative_mask = torch.from_numpy(ndi.binary_fill_holes(notditectedlesion).astype(int))
                false_positive_mask = torch.from_numpy(ndi.binary_fill_holes(wronglesiondetected).astype(int))

                imgs_to_save.append(img_utils.normalize(inputs[j].cpu()))
                t = targets[j].cpu().float().unsqueeze(0)

                imgs_to_save.append(torch.cat((t, t, t), 0))
                np_pred = np_pred.float().unsqueeze(0)
                imgs_to_save.append(torch.cat((np_pred, np_pred, np_pred), 0))

                false_negative_mask = false_negative_mask.float().unsqueeze(0)
                imgs_to_save.append(torch.cat((false_negative_mask, false_negative_mask, false_negative_mask), 0))

                false_positive_mask = false_positive_mask.float().unsqueeze(0)
                imgs_to_save.append(torch.cat((false_positive_mask, false_positive_mask, false_positive_mask), 0))

                img_fpath = os.path.join(output_path, str(image_idx) + '.png')

                save_image(imgs_to_save, img_fpath, nrow=5)

                # save single images
                output_path_img_separate = str(output_path) + '\separate_imgs'
                os.makedirs(output_path_img_separate, exist_ok=True)

                save_image(img_utils.normalize(inputs[j].cpu()),
                           os.path.join(output_path_img_separate, str(image_idx) + '_slice.png'))
                save_image(torch.cat((t, t, t), 0),
                           os.path.join(output_path_img_separate, str(image_idx) + '_ground_truth.png'))
                save_image(torch.cat((np_pred, np_pred, np_pred), 0),
                           os.path.join(output_path_img_separate, str(image_idx) + '_pred.png'))
                save_image(torch.cat((false_negative_mask, false_negative_mask, false_negative_mask), 0),
                           os.path.join(output_path_img_separate, str(image_idx) + '_FN.png'))
                save_image(torch.cat((false_positive_mask, false_positive_mask, false_positive_mask), 0),
                           os.path.join(output_path_img_separate, str(image_idx) + '_FP.png'))

                image_idx = image_idx + 1

                tmp_tp, tmp_fp, tmp_fn, tmp_tn = compute_performance(pred[j], targets[j].cpu())

                test_tp += tmp_tp
                test_fp += tmp_fp
                test_fn += tmp_fn
                test_tn += tmp_tn

    dsc = dice(test_tp, test_fp, test_fn)
    sens = test_tp / (test_tp + test_fn)
    spec = test_tn / (test_tn + test_fp)
    acc = (test_tp + test_tn) / (test_tp + test_tn + test_fn + test_fp)
    error = (test_fp + test_fn) / (test_tp + test_tn + test_fn + test_fp)
    ppv = test_tp / (test_tp + test_fp)
    npv = test_tn / (test_tn + test_fn)
    extra_fraction = (test_fp) / (test_tn + test_fn)
    iou = test_tp / (test_tp + test_fn + test_fp)

    return dsc, sens, spec, acc, error, ppv, npv, extra_fraction, iou
