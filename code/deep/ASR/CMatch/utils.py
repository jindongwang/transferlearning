import torch
import logging
from espnet.asr.asr_utils import add_results_to_json
import argparse
import numpy as np
import collections
import json
def str2bool(str):
	return True if str.lower() == 'true' else False
def setup_logging(verbose=1):
    if verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

# Training stats
def dict_average(dic):
    avg_key, avg_val = [], []
    for key, lst in dic.items():
        if key.endswith("_lst"):
            avg_key.append(key[:-4])
            avg_val.append(np.mean(lst))
    for key, val in zip(avg_key, avg_val):
        dic[key] = val
    return dic

# Load and save
def load_pretrained_model(model, model_path, modules_to_load=None, exclude_modules=None):
    '''
    load_pretrained_model(model=model, model_path="", 
                                modules_to_load=None, exclude_modules="")
    '''
    model_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    if exclude_modules:
        for e in exclude_modules.split(","):
            model_dict = {k: v for k, v in model_dict.items() if not k.startswith(e)}

    if not modules_to_load:
        src_dict = model_dict
    else:
        src_dict = {}
        for module in modules_to_load.split(","):
            src_dict.update({k: v for k, v in model_dict.items() if k.startswith(module)})
    
    dst_state = model.state_dict()
    dst_state.update(src_dict)
    model.load_state_dict(dst_state)
def torch_save(model, save_path, optimizer=None, local_rank=0):
    if local_rank != 0:
        return
    if hasattr(model, "module"):
        state_dict = model.module.state_dict() if not optimizer else collections.OrderedDict(model=model.module.state_dict(), optimizer=optimizer.state_dict())
    else:
        state_dict = model.state_dict() if not optimizer else collections.OrderedDict(model=model.state_dict(), optimizer=optimizer.state_dict())
    torch.save(state_dict, save_path)
def torch_load(snapshot_path, model, optimizer=None):
    # load snapshot
    snapshot_dict = torch.load(snapshot_path, map_location=lambda storage, loc: storage)
    if not "model" in snapshot_dict.keys():
        model_dict = snapshot_dict
        snapshot_dict = collections.OrderedDict(model=model_dict)
    if hasattr(model, "module"):
        model.module.load_state_dict(snapshot_dict["model"])
    else:
        model.load_state_dict(snapshot_dict["model"])
    if optimizer:
        optimizer.load_state_dict(snapshot_dict["optimizer"])
    del snapshot_dict

# Decoding
def compute_wer(ref, hyp, normalize=False):
    """Compute Word Error Rate.
        [Reference]
            https://martin-thoma.com/word-error-rate-calculation/
    Args:
        ref (list): words in the reference transcript
        hyp (list): words in the predicted transcript
        normalize (bool, optional): if True, divide by the length of ref
    Returns:
        wer (float): Word Error Rate between ref and hyp
        n_sub (int): the number of substitution
        n_ins (int): the number of insertion
        n_del (int): the number of deletion
    """
    # Initialisation
    d = np.zeros((len(ref) + 1) * (len(hyp) + 1), dtype=np.uint16)
    d = d.reshape((len(ref) + 1, len(hyp) + 1))
    for i in range(len(ref) + 1):
        for j in range(len(hyp) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # Computation
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                sub_tmp = d[i - 1][j - 1] + 1
                ins_tmp = d[i][j - 1] + 1
                del_tmp = d[i - 1][j] + 1
                d[i][j] = min(sub_tmp, ins_tmp, del_tmp)

    wer = d[len(ref)][len(hyp)]

    # Find out the manipulation steps
    x = len(ref)
    y = len(hyp)
    error_list = []
    while True:
        if x == 0 and y == 0:
            break
        else:
            if x > 0 and y > 0:
                if d[x][y] == d[x - 1][y - 1] and ref[x - 1] == hyp[y - 1]:
                    error_list.append("C")
                    x = x - 1
                    y = y - 1
                elif d[x][y] == d[x][y - 1] + 1:
                    error_list.append("I")
                    y = y - 1
                elif d[x][y] == d[x - 1][y - 1] + 1:
                    error_list.append("S")
                    x = x - 1
                    y = y - 1
                else:
                    error_list.append("D")
                    x = x - 1
            elif x == 0 and y > 0:
                if d[x][y] == d[x][y - 1] + 1:
                    error_list.append("I")
                    y = y - 1
                else:
                    error_list.append("D")
                    x = x - 1
            elif y == 0 and x > 0:
                error_list.append("D")
                x = x - 1
            else:
                raise ValueError

    n_sub = error_list.count("S")
    n_ins = error_list.count("I")
    n_del = error_list.count("D")
    n_cor = error_list.count("C")

    assert wer == (n_sub + n_ins + n_del)
    assert n_cor == (len(ref) - n_sub - n_del)

    if normalize:
        wer /= len(ref)
    return wer, n_sub, n_ins, n_del, n_cor
def recognize_and_evaluate(dataloader, model, args, model_path=None, wer=False, write_to_json=False):
    if model_path:
        torch_load(model_path, model)
    orig_model = model
    if hasattr(model, "module"):
        model = model.module
    if write_to_json:
        # read json data
        assert args.result_label and args.recog_json
        with open(args.recog_json, "rb") as f:
            js = json.load(f)["utts"]
            new_js = {}
    model.eval()
    recog_args = {
        "beam_size": args.beam_size,
        "penalty": args.penalty,
        "ctc_weight": args.ctc_weight,
        "maxlenratio": args.maxlenratio,
        "minlenratio": args.minlenratio,
        "lm_weight": args.lm_weight,
        "rnnlm": args.rnnlm,
        "nbest": args.nbest,
        "space": args.sym_space,
        "blank": args.sym_blank,
    }
    recog_args = argparse.Namespace(**recog_args)

    #progress_bar = tqdm(dataloader)
    #progress_bar.set_description("Testing CER/WERs")
    err_dict = (
        dict(cer=None)
        if not wer
        else dict(cer=collections.defaultdict(int), wer=collections.defaultdict(int))
    )
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            logging.warning(f"Testing CER/WERs: {batch_idx+1}/{len(dataloader)}")
            fbank, ilens, tokens = data
            fbanks = []
            for i, fb in enumerate(fbank):
                fbanks.append(fb[: ilens[i], :])
            fbank = fbanks
            nbest_hyps = model.recognize_batch(
               fbank, recog_args, char_list=None, rnnlm=None
            )
            y_hats = [nbest_hyp[0]["yseq"][1:-1] for nbest_hyp in nbest_hyps]
            if write_to_json:
                for utt_idx in range(len(fbank)):
                    name = dataloader.dataset[batch_idx][utt_idx][0]
                    new_js[name] = add_results_to_json(
                        js[name], nbest_hyps[utt_idx], args.char_list
                    )
            for i, y_hat in enumerate(y_hats):
                y_true = tokens[i]

                hyp_token = [
                    args.char_list[int(idx)] for idx in y_hat if int(idx) != -1
                ]
                ref_token = [
                    args.char_list[int(idx)] for idx in y_true if int(idx) != -1
                ]
                for key in sorted(err_dict.keys()):  # cer then wer
                    if key == "wer":
                        if args.bpemodel:
                            ref_token = args.bpemodel.decode_pieces(ref_token).split()
                            hyp_token = args.bpemodel.decode_pieces(hyp_token).split()
                        else:
                            ref_token = (
                                " ".join(ref_token)
                                .replace(" ", "")
                                .replace("<space>", " ")
                                .split()
                            )  # sclite does not consider the number of spaces when splitting
                            hyp_token = (
                                " ".join(hyp_token)
                                .replace(" ", "")
                                .replace("<space>", " ")
                                .split()
                            )
                        logging.debug("HYP: " + str(hyp_token))
                        logging.debug("REF: " + str(ref_token))
                    utt_err, utt_nsub, utt_nins, utt_ndel, utt_ncor = compute_wer(
                        ref_token, hyp_token
                    )

                    err_dict[key]["n_word"] += len(ref_token)
                    if utt_err != 0:
                        err_dict[key]["n_err"] += utt_err  # Char / word error
                        err_dict[key]["n_ser"] += 1  # Sentence error
                    err_dict[key]["n_cor"] += utt_ncor
                    err_dict[key]["n_sub"] += utt_nsub
                    err_dict[key]["n_ins"] += utt_nins
                    err_dict[key]["n_del"] += utt_ndel
                    err_dict[key]["n_sent"] += 1

    for key in err_dict.keys():
        err_dict[key]["err"] = err_dict[key]["n_err"] / err_dict[key]["n_word"] * 100.0
        err_dict[key]["ser"] = err_dict[key]["n_ser"] / err_dict[key]["n_word"] * 100.0
    torch.cuda.empty_cache()
    if write_to_json:
        with open(args.result_label, "wb") as f:
            f.write(
                json.dumps(
                    {"utts": new_js}, indent=4, ensure_ascii=False, sort_keys=True
                ).encode("utf_8")
            )
    model = orig_model
    return err_dict