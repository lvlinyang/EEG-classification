import logging
import os.path
import time
from collections import OrderedDict
import sys
import torch
from braindecode.datasets.bcic_iv_2a import BCICompetition4Set2A
from braindecode.torch_ext.util import set_random_seeds
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.datautil.signalproc import (
    bandpass_cnt,
    exponential_running_standardize,
)
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne
from braindecode.models.self_shallow import SelfShallow
from dataprocessing.wave_transform import *
log = logging.getLogger(__name__)


def run_exp(data_folder, subject_id, low_cut_hz, test_model, model_PATH, model, cuda):
    ival = [-500, 4000]
    high_cut_hz = 38
    factor_new = 1e-3
    init_block_size = 1000

    test_filename = "A{:02d}E.gdf".format(subject_id)
    test_filepath = os.path.join(data_folder, test_filename)
    test_label_filepath = test_filepath.replace(".gdf", ".mat")

    test_loader = BCICompetition4Set2A(
        test_filepath, labels_filename=test_label_filepath
    )
    test_cnt = test_loader.load()
    # Preprocessing
    test_cnt = test_cnt.drop_channels(["EOG-left", "EOG-central", "EOG-right"])
    assert len(test_cnt.ch_names) == 22
    test_cnt = mne_apply(lambda a: a * 1e6, test_cnt)
    test_cnt = mne_apply(
        lambda a: bandpass_cnt(
            a,
            low_cut_hz,
            high_cut_hz,
            test_cnt.info["sfreq"],
            filt_order=3,
            axis=1,
        ),
        test_cnt,
    )
    test_cnt = mne_apply(
        lambda a: exponential_running_standardize(
            a.T,
            factor_new=factor_new,
            init_block_size=init_block_size,
            eps=1e-4,
        ).T,
        test_cnt,
    )
    marker_def = OrderedDict(
        [
            ("Left Hand", [1]),
            ("Right Hand", [2]),
            ("Foot", [3]),
            ("Tongue", [4]),
        ]
    )
    test_set = create_signal_target_from_raw_mne(test_cnt, marker_def, ival)
    test_set = data_all_chan_cwtandraw(test_set)
    set_random_seeds(seed=20200104, cuda=cuda)

    model = SelfShallow()

    if test_model:
        model.load_state_dict(torch.load(model_PATH))

    if cuda:
        model.cuda()
        model.eval()
    log.info("Model: \n{:s}".format(str(model)))
    all_test_labels = test_set.y
    all_test_data = torch.from_numpy(test_set.X).cuda()
    preds,feature1,feature2,raw,guide  = model(all_test_data)

    preds = preds.cpu()
    preds = preds.detach().numpy()
    all_preds = np.argmax(preds, axis=1).squeeze()
    accury = np.mean(all_test_labels==all_preds)
    print('accury:',accury)

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s : %(message)s",
        level=logging.DEBUG,
        stream=sys.stdout,
    )
    # Should contain both .gdf files and .mat-labelfiles from competition
    data_folder = '/home/user/lly/BCICIV_2a_gdf'

    # subject_id = 9  # 1-9
    subject = 3
    low_cut_hz = 0
    model = "shallow_allchan"  #'shallow_allchan',
    cuda = True
    # torch.backends.cudnn.deterministic = True
    test_model = True
    model_PATH ='/home/user/lly/my_github_model/best_models/case_1/3-1538-0.045139.pkl'


    exp = run_exp(data_folder, subject, low_cut_hz, test_model, model_PATH, model, cuda)
