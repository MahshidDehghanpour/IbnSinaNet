import argparse
import torch.optim as torch_optim


def process_args():
    opt_defs = {
        "n_classes":                dict(flags=('-nc',     '--nclasses'),
                                         info=dict(default=2, type=int, help="num of classes")),
        "mean":                     dict(flags=('-mean',   '--mean'),
                                         info=dict(default=0.1026, type=float, help="mean")),
        "std":                      dict(flags=('-std',    '--std'),
                                         info=dict(default=0.0971, type=float, help="std")),
        "dataset_path":             dict(flags=('-dp',     '--dataset-path'),
                                         info=dict(default="/home/user/mahcod/datasets/ISBI_2015",
                                                   type=str, help="path to dataset")),
        "val_dataset":              dict(flags=('-vd',     '--val-dataset'),
                                         info=dict(default='val', type=str, help="val or test")),
        "folders":                  dict(flags=('-f',      '--folders'),
                                         info=dict(default=5, type=int, help="num folders")),
        "use_sa":                   dict(flags=('-usesa',  '--use-sa'),
                                         info=dict(default=True, type=bool, help="use Squeeze Attention")),
        "use_stn":                  dict(flags=('-usestn', '--use-stn'),
                                         info=dict(default=False, type=bool, help="use STN")),
        "use_lstm":                 dict(flags=('-uselstm','--use-lstm'),
                                         info=dict(default=False, type=bool, help="use LSTM")),
        "lstm_kernel_size":         dict(flags=('-lks',    '--lstm-kernel-size'),
                                         info=dict(default=3, type=int, help="lstm kernel size")),
        "lstm_num_layers":          dict(flags=('-lnl',    '--lstm-num-layers'),
                                         info=dict(default=1, type=int, help="lstm num layers")),
        "seq_size":                 dict(flags=('-ss',     '--seq-size'),
                                         info=dict(default=3, type=int, help="sequence size")),
        "sliding_window":           dict(flags=('-sw',     '--sliding-window'),
                                         info=dict(default=False, type=bool, help="sliding window")),
        "bidirectional":            dict(flags=('-bi',     '--bidirectional'),
                                         info=dict(default=False, type=bool, help="bidirectional")),
        "input_dim":                dict(flags=('-dim',    '--input-dim'),
                                         info=dict(default=160, type=int, help="input dim")),
        # batch_size per GPU — با 6 GPU میشه effective batch=12
        "batch_size":               dict(flags=('-b',      '--batch-size'),
                                         info=dict(default=2, type=int, help="batch size per GPU")),
        "optim":                    dict(flags=('-opt',    '--optim'),
                                         info=dict(default='Adam', type=str, help="optimizer")),
        "learning_rate":            dict(flags=('-lr',     '--learning-rate'),
                                         info=dict(default=3e-4, type=float, help="learning rate")),
        "learning_rate_decay_by":   dict(flags=('-lrdb',  '--learning-rate-decay-by'),
                                         info=dict(default=0.99, type=float, help="lr decay factor")),
        "learning_rate_decay_every":dict(flags=('-lrde',  '--learning-rate-decay-every'),
                                         info=dict(default=1, type=int, help="lr decay every n epochs")),
        "weight_decay":             dict(flags=('-wd',     '--weight-decay'),
                                         info=dict(default=3e-4, type=float, help="weight decay")),
        "num_epochs":               dict(flags=('-ne',     '--num-epochs'),
                                         info=dict(default=150, type=int, help="training epochs")),
        "dropout":                  dict(flags=('-drop',   '--dropout'),
                                         info=dict(default=0.1, type=float, help="dropout")),
        "drop_last":                dict(flags=('-dl',     '--drop-last'),
                                         info=dict(default=True, type=bool, help="drop last")),
        "results_path":             dict(flags=('-rp',     '--results-path'),
                                         info=dict(default="./results_ms/", type=str, help="results path")),
        "weights_path":             dict(flags=('-wp',     '--weights-path'),
                                         info=dict(default="./weights/", type=str, help="weights path")),
        "weights_fname":            dict(flags=('-wf',     '--weights-fname'),
                                         info=dict(default=None, type=str, help="weights filename")),
        "last_tag":                 dict(flags=('-tag',    '--last-tag'),
                                         info=dict(default=0, type=int, help="last tag")),
    }

    parser = argparse.ArgumentParser()
    for k, arg in opt_defs.items():
        parser.add_argument(*arg["flags"], **arg["info"])
    opt = parser.parse_args(None)

    # ── تنظیمات ثابت مدل ──────────────────────────────────────────────────
    opt.down_blocks = [5, 5, 5, 5, 5]
    opt.up_blocks   = [5, 5, 5, 5]
    opt.down_layers = [4, 4, 4, 4, 4]
    opt.up_layers   = [4, 4, 4, 4]
    opt.activation  = "Relu"
    opt.group_multi = 2

    # ── optimizer ──────────────────────────────────────────────────────────
    opt.optimizer = torch_optim.Adam
    opt.lr_decay  = opt.learning_rate_decay_by

    # ── ablation defaults ──────────────────────────────────────────────────
    opt.ablation_csp  = True
    opt.ablation_gcnn = True

    # ── loss function ──────────────────────────────────────────────────────
    opt.loss_type = 'CE+Dice'

    return opt
