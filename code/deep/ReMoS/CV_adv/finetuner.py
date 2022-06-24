import os
import os.path as osp
import time
from functools import partial
import torch
from advertorch.attacks import LinfPGDAttack

from dataset.cub200 import CUB200Data
from dataset.mit67 import MIT67Data
from dataset.stanford_dog import SDog120Data
from dataset.stanford_40 import Stanford40Data
from dataset.flower102 import Flower102Data

from model.fe_resnet import resnet18_dropout, resnet34_dropout, resnet50_dropout, resnet101_dropout
from model.fe_resnet import feresnet18, feresnet34, feresnet50, feresnet101

from eval_robustness import advtest, myloss
from utils import *

class Finetuner(object):
    def __init__(
        self,
        args,
        model,
        teacher,
        train_loader,
        test_loader,
    ):
        self.args = args
        self.model = model
        self.teacher = teacher
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.init_models()

    def init_models(self):
        args = self.args

        # Adv eval
        eval_pretrained_model = eval('fe{}'.format(args.network))(pretrained=True).eval().cuda()
        adversary = LinfPGDAttack(
                eval_pretrained_model, loss_fn=myloss, eps=args.B,
                nb_iter=args.pgd_iter, eps_iter=0.01, 
                rand_init=True, clip_min=-2.2, clip_max=2.2,
                targeted=False)
        adveval_test_loader = torch.utils.data.DataLoader(
            self.test_loader.dataset,
            batch_size=8, shuffle=False,
            num_workers=8, pin_memory=False
        )
        self.adv_eval_fn = partial(
            advtest,
            loader=adveval_test_loader,
            adversary=adversary,
            args=args,
        )

    def adv_eval(self):
        model = self.model
        args = self.args
        clean_top1, adv_top1, adv_sr = self.adv_eval_fn(model)
        result_sum = f'Clean Top-1: {clean_top1:.2f} | Adv Top-1: {adv_top1:.2f} | Attack Success Rate: {adv_sr:.2f}'
        with open(osp.join(args.output_dir, "posttrain_eval.txt"), "w") as f:
            f.write(result_sum)

    def compute_loss(
        self, batch, label, ce, 
    ):
        model = self.model
        out = model(batch)
        _, pred = out.max(dim=1)

        top1 = float(pred.eq(label).sum().item()) / label.shape[0] * 100.
        loss = ce(out, label)
        return loss, top1

    def test(self, loader):
        model = self.model        
        with torch.no_grad():
            model.eval()
            ce = CrossEntropyLabelSmooth(loader.dataset.num_classes)

            total_ce = 0
            total = 0
            top1 = 0
            for i, (batch, label) in enumerate(loader):
                batch, label = batch.to('cuda'), label.to('cuda')
                total += batch.size(0)
                out = model(batch)
                _, pred = out.max(dim=1)
                top1 += int(pred.eq(label).sum().item())
                total_ce += ce(out, label).item()

        return float(top1)/total*100, total_ce/(i+1)
    
    def train(self):
        model = self.model
        train_loader = self.train_loader
        iterations = self.args.iterations
        lr = self.args.lr
        output_dir = self.args.output_dir
        teacher = self.teacher
        args = self.args
        model = model.to('cuda')
        
        fc_module = model.fc
        fc_params = list(map(id, fc_module.parameters()))
        base_params = filter(lambda p: id(p) not in fc_params,
                        self.model.parameters())
        optimizer = torch.optim.SGD(
            [
                {'params': base_params},
                {'params': fc_module.parameters(), 'lr': lr*10}
            ], 
            lr=lr, 
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

        teacher.eval()
        ce = CrossEntropyLabelSmooth(train_loader.dataset.num_classes)

        batch_time = MovingAverageMeter('Time', ':6.3f')
        data_time = MovingAverageMeter('Data', ':6.3f')
        ce_loss_meter = MovingAverageMeter('CE Loss', ':6.3f')
        top1_meter  = MovingAverageMeter('Acc@1', ':6.2f')

        train_path = osp.join(output_dir, "train.tsv")
        with open(train_path, 'w') as wf:
            columns = ['time', 'iter', 'Acc', 'celoss']
            wf.write('\t'.join(columns) + '\n')
        test_path = osp.join(output_dir, "test.tsv")
        with open(test_path, 'w') as wf:
            columns = ['time', 'iter', 'Acc', 'celoss']
            wf.write('\t'.join(columns) + '\n')
        adv_path = osp.join(output_dir, "adv.tsv")
        with open(adv_path, 'w') as wf:
            columns = ['time', 'iter', 'Acc', 'AdvAcc', 'ASR']
            wf.write('\t'.join(columns) + '\n')
        
        dataloader_iterator = iter(train_loader)
        for i in range(iterations):
            model.train()
            optimizer.zero_grad()

            end = time.time()
            try:
                batch, label = next(dataloader_iterator)
            except:
                dataloader_iterator = iter(train_loader)
                batch, label = next(dataloader_iterator)
            batch, label = batch.to('cuda'), label.to('cuda')
            data_time.update(time.time() - end)

            loss, top1 = self.compute_loss(
                batch, label, ce, 
            )

            top1_meter.update(top1)
            ce_loss_meter.update(loss)
            
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)

            if (i % args.print_freq == 0) or (i == iterations-1):
                progress = ProgressMeter(
                    iterations,
                    [batch_time, data_time, top1_meter, ce_loss_meter],
                    prefix="PID {} ".format(self.args.pid),
                    output_dir=output_dir,
                )
                progress.display(i)

            if (i % args.test_interval == 0) or (i == iterations-1):
                test_top1, test_ce_loss = self.test(self.train_loader)
                train_top1, train_ce_loss = self.test(self.test_loader)
                print(
                    'Eval Train | Iteration {}/{} | Top-1: {:.2f} | CE Loss: {:.3f} | PID {}'.format(i+1, iterations, train_top1, train_ce_loss, self.args.pid))
                print(
                    'Eval Test | Iteration {}/{} | Top-1: {:.2f} | CE Loss: {:.3f} | PID {}'.format(i+1, iterations, test_top1, test_ce_loss, self.args.pid))
                localtime = time.asctime( time.localtime(time.time()) )[4:-6]
                with open(train_path, 'a') as af:
                    train_cols = [
                        localtime,
                        i, 
                        round(train_top1,2), 
                        round(train_ce_loss,2), 
                    ]
                    af.write('\t'.join([str(c) for c in train_cols]) + '\n')
                with open(test_path, 'a') as af:
                    test_cols = [
                        localtime,
                        i, 
                        round(test_top1,2), 
                        round(test_ce_loss,2), 
                    ]
                    af.write('\t'.join([str(c) for c in test_cols]) + '\n')

                ckpt_path = osp.join(
                    args.output_dir,
                    "ckpt.pth"
                )
                torch.save(
                    {'state_dict': model.state_dict()}, 
                    ckpt_path,
                )

            if ( hasattr(self, "iterative_prune") and i % args.prune_interval == 0 ):
                self.iterative_prune(i)

            if ( 
                args.adv_test_interval > 0 and 
                ( (i % args.adv_test_interval == 0) or (i == iterations-1) )
            ):
                clean_top1, adv_top1, adv_sr = self.adv_eval_fn(model)
                localtime = time.asctime( time.localtime(time.time()) )[4:-6]
                with open(adv_path, 'a') as af:
                    test_cols = [
                        localtime,
                        i, 
                        round(clean_top1,2),
                        round(adv_top1,2),
                        round(adv_sr,2),
                    ]
                    af.write('\t'.join([str(c) for c in test_cols]) + '\n')

        return model
