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
from balanced_sampler import BalancedBatchSampler
#cv mt cnh ky dv sl el lv fy-NL sah
data_config = {
    "template100": {
        "train": "dump/train_template/deltafalse/data_unigram100.json",
        "val": "dump/dev_template/deltafalse/data_unigram100.json",
        "test": "dump/test_template/deltafalse/data_unigram100.json",
        "token": "data/template_lang_char/train_template_unigram100_units.txt",
        "prefix": "/D_data/commonvoice/asr1/",
        "bpemodel": "data/template_lang_char/train_template_unigram100.model",
    },
    "template150": {
        "train": "dump/train_template/deltafalse/data_unigram150.json",
        "val": "dump/dev_template/deltafalse/data_unigram150.json",
        "test": "dump/test_template/deltafalse/data_unigram150.json",
        "token": "data/template_lang_char/train_template_unigram150_units.txt",
        "prefix": "/D_data/commonvoice/asr1/",
        "bpemodel": "data/template_lang_char/train_template_unigram150.model",
    },
}
low_resource_languages = ["ro", "cs", "br", "ar", "uk"]

def read_json_file(fname):
    with open(fname, "rb") as f:
        contents = json.load(f)["utts"]
    return contents
def load_json(train_json_file, dev_json_file, test_json_file):
    train_json = read_json_file(train_json_file)
    if os.path.isfile(dev_json_file):
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
    language = dataset
    if language in low_resource_languages:
        template_key = "template100"
    else:
        template_key = "template150"
    data_config[dataset] = data_config[template_key].copy()   
    for key in ["train", "val", "test", "token"]:
        data_config[dataset][key] = data_config[template_key][key].replace("template", dataset)
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
    # trainset = make_batchset(train_json, batch_size, max_length_in=800, max_length_out=150)
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
        args.batch_size if args.ngpu <= 1 else int(args.batch_size / args.ngpu),
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
        args.batch_size if args.ngpu <= 1 else int(args.batch_size / args.ngpu),
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


def load_multilingual_data(root_path, datasets, args, languages):
    def collate(minibatch):
        out = []
        for b in minibatch:
            fbanks = []
            tokens = []
            language = None
            for _, info in b:
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
                if language is not None:
                    assert language == info['category']
                else:
                    language = info['category']
            ilens = torch.tensor([x.shape[0] for x in fbanks])
            out.append((
            pad_sequence(fbanks, batch_first=True, padding_value=0),
            ilens,
            pad_sequence(tokens, batch_first=True, padding_value=-1),
            language,
        ))
        return out[0] if len(out) == 1 else out
    idim = None
    odim_dict = {}
    mtl_train_json, mtl_dev_json, mtl_test_json = {}, {}, {}
    for idx, dataset in enumerate(datasets):
        language = dataset
        if language in low_resource_languages:
            template_key = "template100"
        else:
            template_key = "template150"
        data_config[dataset] = data_config[template_key].copy()   
        for key in ["train", "val", "test", "token"]:
            data_config[dataset][key] = data_config[template_key][key].replace("template", dataset)
        
        train_json = os.path.join(root_path, data_config[dataset]["train"])
        dev_json = (
            os.path.join(root_path, data_config[dataset]["val"])
            if data_config[dataset]["val"]
            else f"{root_path}/tmp_dev_set_{dataset}.json"
        )
        test_json = os.path.join(root_path, data_config[dataset]["test"])
        train_json, dev_json, test_json = load_json(train_json, dev_json, test_json)
        for key in train_json.keys():
            train_json[key]['category'] = language
        for key in dev_json.keys():
            dev_json[key]['category'] = language
        for key in test_json.keys():
            test_json[key]['category'] = language
        #print(train_json)
        _, info = next(iter(train_json.items()))
        if idim is not None:
            assert idim == info["input"][0]["shape"][1]
        else:
            idim = info["input"][0]["shape"][1]
        odim_dict[language] = info["output"][0]["shape"][1]
        
        # Break if not in specified languages
        if dataset not in languages:
            continue

        mtl_train_json.update(train_json)
        mtl_dev_json.update(dev_json)
        mtl_test_json.update(test_json)
        #print(len(mtl_train_json), len(train_json))
    train_json, dev_json, test_json = mtl_train_json, mtl_dev_json, mtl_test_json
    use_sortagrad = False  # args.sortagrad == -1 or args.sortagrad > 0
    # trainset = make_batchset(train_json, batch_size, max_length_in=800, max_length_out=150)
    if args.ngpu > 1 and not args.dist_train:
        min_batch_size = args.ngpu
    else:
        min_batch_size = 1
    if args.meta_train:
        min_batch_size = 2 * min_batch_size
    trainset = make_batchset(
        train_json,
        args.batch_size,
        args.maxlen_in,
        args.maxlen_out,
        args.minibatches,
        min_batch_size=min_batch_size,
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
        args.batch_size if args.ngpu <= 1 else int(args.batch_size / args.ngpu),
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
        args.batch_size if args.ngpu <= 1 else int(args.batch_size / args.ngpu),
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
    elif args.meta_train:
        train_sampler = BalancedBatchSampler(trainset)
    else:
        train_sampler = None
    train_loader = DataLoader(
        trainset,
        batch_size=1 if not args.meta_train else len(languages),
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
    return (train_loader, dev_loader, test_loader), (idim, odim_dict)

def load_token_list(token_file):
    with open(token_file, "r") as f:
        token_list = [entry.split()[0] for entry in f]
    token_list.insert(0, "<blank>")
    token_list.append("<eos>")
    return token_list

def load_bpemodel(root_path, dataset):
    if dataset in low_resource_languages:
        template_key = "template100"
    else:
        template_key = "template150"
    bpemodel_path = os.path.join(root_path, data_config[template_key]["bpemodel"]).replace("template", dataset)
    bpemodel = spm.SentencePieceProcessor()
    bpemodel.Load(bpemodel_path)
    return bpemodel
