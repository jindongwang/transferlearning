from espnet.utils.training.batchfy import make_batchset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import os
import json
import kaldiio
import random
import logging
import sentencepiece as spm

data_config = {
    "librispeech": {
        "train": "dump/train_960/deltafalse/data_unigram5000.json",
        "val": "dump/dev_clean/deltafalse/data_unigram5000.json",
        "test": "dump/test_clean/deltafalse/data_unigram5000.json",
        "token": "data/lang_char/train_960_unigram5000_units.txt",
        "prefix": "/espnet/egs/librispeech/asr1",
    },
    "wsj": {
        "train": "dump/train_si284/deltafalse/data.json",
        "val": "dump/test_dev93/deltafalse/data.json",
        "test": "dump/test_eval92/deltafalse/data.json",
        "token": "data/lang_1char/train_si284_units.txt",
        "prefix": "/opt/espnet/egs/wsj/asr1",
    },
    "an4": {
        "train": "dump/train_nodev/deltafalse/data.json",
        "val": "dump/train_dev/deltafalse/data.json",
        "test": "dump/test/deltafalse/data.json",
        "token": "data/lang_1char/train_nodev_units.txt",
        "prefix": "/home/jindwang/mine/espnet/egs/an4/asr1",
    },
    "libriadapt_en_us_clean_matrix": {
        "train": "dump/en_us_clean_matrix/train/deltafalse/data_unigram31.json",
        "val": None,
        "test": "dump/en_us_clean_matrix/test/deltafalse/data_unigram31.json",
        "token": "data/lang_char/en_us_clean_matrix/train_unigram31_units.txt",
        "prefix": "/D_data/libriadapt_processed/asr1",
        "bpemodel": "data/lang_char/en_us_clean_matrix/train_unigram31.model",
    },
    "libriadapt_en_us_clean_shure": {
        "train": "dump/en_us_clean_shure/train/deltafalse/data_unigram31.json",
        "val": None,
        "test": "dump/en_us_clean_shure/test/deltafalse/data_unigram31.json",
        "token": "data/lang_char/en_us_clean_shure/train_unigram31_units.txt",
        "prefix": "/D_data/libriadapt_processed/asr1",
        "bpemodel": "data/lang_char/en_us_clean_shure/train_unigram31.model",
    },
    "libriadapt_en_gb_clean_shure": {
        "train": "dump/en_gb_clean_shure/train/deltafalse/data_unigram31.json",
        "val": None,
        "test": "dump/en_gb_clean_shure/test/deltafalse/data_unigram31.json",
        "token": "data/lang_char/en_gb_clean_shure/train_unigram31_units.txt",
        "prefix": "/D_data/libriadapt_processed/asr1",
        "bpemodel": "data/lang_char/en_gb_clean_shure/train_unigram31.model",
    },
    "libriadapt_en_in_clean_shure": {
        "train": "dump/en_in_clean_shure/train/deltafalse/data_unigram31.json",
        "val": None,
        "test": "dump/en_in_clean_shure/test/deltafalse/data_unigram31.json",
        "token": "data/lang_char/en_in_clean_shure/train_unigram31_units.txt",
        "prefix": "/D_data/libriadapt_processed/asr1",
        "bpemodel": "data/lang_char/en_in_clean_shure/train_unigram31.model",
    },
}


def read_json_file(fname):
    with open(fname, "rb") as f:
        contents = json.load(f)["utts"]
    return contents


def load_json(train_json_file, dev_json_file, test_json_file):
    train_json = read_json_file(train_json_file)
    if os.path.isfile(dev_json_file) and not "tmp_dev_set" in dev_json_file:
        dev_json = read_json_file(dev_json_file)
    else:
        n_samples = len(train_json)
        train_size = int(0.9 * n_samples)
        logging.warning(
            f"No dev set provided, will split the last {n_samples - train_size} (10%) samples from training data"
        )
        train_json_item = list(train_json.items())
        # random.shuffle(train_json_item)
        train_json = dict(train_json_item[:train_size])
        dev_json = dict(train_json_item[train_size:])
        # Save temp dev set
        with open(dev_json_file, "w") as f:
            json.dump({"utts": dev_json}, f)
        logging.warning(f"Temporary dev set saved: {dev_json_file}")
    test_json = read_json_file(test_json_file)
    logging.warning(f"#Train Json {train_json_file}: {len(train_json)}")
    logging.warning(f"#Dev Json {dev_json_file}: {len(dev_json)}")
    logging.warning(f"#Test Json {test_json_file}: {len(test_json)}")
    return train_json, dev_json, test_json


def load_data(root_path, dataset, args):
    def collate(minibatch):
        fbanks = []
        tokens = []
        for _, info in minibatch[0]:
            fbanks.append(
                torch.tensor(
                    kaldiio.load_mat(
                        info["input"][0]["feat"].replace(
                            data_config[dataset]["prefix"], root_path
                        )
                    )
                )
            )
            tokens.append(
                torch.tensor([int(s) for s in info["output"][0]["tokenid"].split()])
            )
        ilens = torch.tensor([x.shape[0] for x in fbanks])
        return (
            pad_sequence(fbanks, batch_first=True, padding_value=0),
            ilens,
            pad_sequence(tokens, batch_first=True, padding_value=-1),
        )

    train_json = os.path.join(root_path, data_config[dataset]["train"])
    dev_json = (
        os.path.join(root_path, data_config[dataset]["val"])
        if data_config[dataset]["val"]
        else f"{root_path}/tmp_dev_set_{dataset}.json"
    )
    test_json = os.path.join(root_path, data_config[dataset]["test"])
    train_json, dev_json, test_json = load_json(train_json, dev_json, test_json)
    _, info = next(iter(train_json.items()))
    idim = info["input"][0]["shape"][1]
    odim = info["output"][0]["shape"][1]

    use_sortagrad = False  # args.sortagrad == -1 or args.sortagrad > 0
    trainset = make_batchset(
        train_json,
        args.batch_size,
        args.maxlen_in,
        args.maxlen_out,
        args.minibatches,
        min_batch_size=args.ngpu if (args.ngpu > 1 and not args.dist_train) else 1,
        shortest_first=use_sortagrad,
        count=args.batch_count,
        batch_bins=args.batch_bins,
        batch_frames_in=args.batch_frames_in,
        batch_frames_out=args.batch_frames_out,
        batch_frames_inout=args.batch_frames_inout,
        iaxis=0,
        oaxis=0,
    )
    # devset = make_batchset(dev_json, batch_size, max_length_in=800, max_length_out=150)
    devset = make_batchset(
        dev_json,
        args.batch_size if (args.ngpu <= 1 or args.dist_train) else int(args.batch_size / args.ngpu),
        args.maxlen_in,
        args.maxlen_out,
        args.minibatches,
        min_batch_size=1,
        count=args.batch_count,
        batch_bins=args.batch_bins,
        batch_frames_in=args.batch_frames_in,
        batch_frames_out=args.batch_frames_out,
        batch_frames_inout=args.batch_frames_inout,
        iaxis=0,
        oaxis=0,
    )
    testset = make_batchset(
        test_json,
        args.batch_size if (args.ngpu <= 1 or args.dist_train) else int(args.batch_size / args.ngpu),
        args.maxlen_in,
        args.maxlen_out,
        args.minibatches,
        min_batch_size=1,
        count=args.batch_count,
        batch_bins=args.batch_bins,
        batch_frames_in=args.batch_frames_in,
        batch_frames_out=args.batch_frames_out,
        batch_frames_inout=args.batch_frames_inout,
        iaxis=0,
        oaxis=0,
    )
    if args.dist_train and args.ngpu > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    else:
        train_sampler = None
    train_loader = DataLoader(
        trainset,
        batch_size=1,
        collate_fn=collate,
        num_workers=args.n_iter_processes,
        shuffle=(train_sampler is None),
        pin_memory=True,
        sampler=train_sampler,
    )
    dev_loader = DataLoader(
        devset,
        batch_size=1,
        collate_fn=collate,
        shuffle=False,
        num_workers=args.n_iter_processes,
        pin_memory=True,
    )
    test_loader = DataLoader(
        testset,
        batch_size=1,
        collate_fn=collate,
        shuffle=False,
        num_workers=args.n_iter_processes,
        pin_memory=True,
    )
    return (train_loader, dev_loader, test_loader), (idim, odim)


def load_token_list(token_file):
    with open(token_file, "r") as f:
        token_list = [entry.split()[0] for entry in f]
    token_list.insert(0, "<blank>")
    token_list.append("<eos>")
    return token_list


def load_bpemodel(root_path, dataset):
    bpemodel_path = os.path.join(root_path, data_config[dataset]["bpemodel"])
    bpemodel = spm.SentencePieceProcessor()
    bpemodel.Load(bpemodel_path)
    return bpemodel
