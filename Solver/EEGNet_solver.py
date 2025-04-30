import os
import numpy as np
import torch
from metrics import cal_log
from utils import print_update, createFolder, write_json, print_dict
import torch.nn as nn

torch.set_printoptions(linewidth=1000)

class Solver:
    def __init__(self, args, net, train_loader, val_loader, test_loader, criterion, optimizer, scheduler, log_dict):
        self.args = args
        self.net = net
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.criterion = criterion
        self.kl_criterion = nn.KLDivLoss(reduction="batchmean")
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.log_dict = log_dict

        self.use_mutual_learning = getattr(args, "use_mutual_learning", False)
        self.use_multi_source_align = getattr(args, "use_multi_source_align", False)
        self.use_kl_alignment = getattr(args, "use_kl_alignment", False)
        self.use_domain_classifier = getattr(args, "use_domain_classifier", False)

        if self.use_multi_source_align:
            self.domain_criterion = torch.nn.CrossEntropyLoss()

        self.best_score = np.inf  # Validation loss 기준
        # 항상 저장 폴더 만들어줌
        createFolder(os.path.join(self.args.save_path, "checkpoint"))

    def compute_loss(self, outputs, labels, domain_labels=None):
        total_loss = 0.0 
        loss_dict = {} 

        loss_cls = self.criterion(outputs["cls"], labels)
        total_loss = loss_cls
        loss_dict["loss_cls"] = loss_cls.item() 

        if self.use_mutual_learning and "freq" in outputs:
            loss_freq = self.criterion(outputs["freq"], labels)
            total_loss += self.args.freq_loss_weight * loss_freq
            loss_dict["loss_freq"] = loss_freq.item()
        
        if self.use_kl_alignment and "cls" in outputs and "freq" in outputs:
            time_pred = torch.log_softmax(outputs["cls"], dim=1)
            freq_pred = torch.softmax(outputs["freq"], dim=1)
            loss_kl = self.kl_criterion(time_pred, freq_pred) + self.kl_criterion(torch.log(freq_pred + 1e-8), torch.softmax(outputs["cls"], dim=1))
            total_loss += self.args.kl_loss_weight * loss_kl
            loss_dict["loss_kl"] = loss_kl.item()

        if self.use_domain_classifier and "domain" in outputs and domain_labels is not None:
            loss_domain = self.domain_criterion(outputs["domain"], domain_labels)
            total_loss += self.args.domain_loss_weight * loss_domain
            loss_dict["loss_domain"] = loss_domain.item()

        return total_loss

    def train(self):
        log_tmp = {key: [] for key in self.log_dict.keys() if "train" in key}
        self.net.train()
        
        for i, data in enumerate(self.train_loader):
            if self.use_multi_source_align:
                inputs, labels, domain_labels = data
                domain_labels = domain_labels.cuda()
            else:
                inputs, labels = data
                domain_labels = None
            inputs = inputs.cuda()
            labels = labels.cuda()

            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.compute_loss(outputs, labels, domain_labels)

            loss.backward()
            self.optimizer.step()

            cal_log(log_tmp, outputs=outputs["cls"], labels=labels, loss=loss)

            sentence = f"({(i + 1) * self.args.batch_size} / {len(self.train_loader.dataset.X)})"
            for key, value in log_tmp.items():
                sentence += f" {key}: {value[i]:0.3f}"
            print_update(sentence, i)
        print("")

        for key in log_tmp.keys():
            self.log_dict[key].append(np.mean(log_tmp[key]))

        val_loss = self.validate()

        # 무조건 checkpoint 저장
        checkpoint_path = os.path.join(self.args.save_path, "checkpoint", "last_model.tar")
        torch.save({
            'net_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None
        }, checkpoint_path)

        # 최고 성능 모델 따로 저장
        if val_loss < self.best_score:
            self.best_score = val_loss
            best_checkpoint_path = os.path.join(self.args.save_path, "checkpoint", "best_model.tar")
            torch.save({
                'net_state_dict': self.net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None
            }, best_checkpoint_path)

    def validate(self):
        log_tmp = {key: [] for key in self.log_dict.keys() if "val" in key}
        self.net.eval()
        total_loss = []

        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                if self.use_multi_source_align:
                    inputs, labels, domain_labels = data
                    domain_labels = domain_labels.cuda()
                else:
                    inputs, labels = data
                    domain_labels = None
                inputs = inputs.cuda()
                labels = labels.cuda()

                outputs = self.net(inputs)
                loss = self.compute_loss(outputs, labels, domain_labels)

                total_loss.append(loss.item())
                cal_log(log_tmp, outputs=outputs["cls"], labels=labels, loss=loss)

        for key in log_tmp.keys():
            self.log_dict[key].append(np.mean(log_tmp[key]))

        return np.mean(total_loss)

    def test(self):
        print("[Start test]")

        checkpoint = torch.load(os.path.join(self.args.save_path, "checkpoint", "best_model.tar"))
        self.net.load_state_dict(checkpoint['net_state_dict'])

        log_tmp = {key: [] for key in self.log_dict.keys() if "test" in key}
        self.net.eval()

        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                if self.use_multi_source_align:
                    inputs, labels, domain_labels = data
                    domain_labels = domain_labels.cuda()
                else:
                    inputs, labels = data
                    domain_labels = None

                inputs = inputs.cuda()
                labels = labels.cuda()

                outputs = self.net(inputs)
                loss = self.compute_loss(outputs, labels, domain_labels)

                cal_log(log_tmp, outputs=outputs["cls"], labels=labels, loss=loss)

        for key in log_tmp.keys():
            self.log_dict[key].append(np.mean(log_tmp[key]))

        print("=> Test Finished")
        for key, value in log_tmp.items():
            print(f"{key}: {np.mean(value):.3f}")

        print("====================================Finish====================================")

    def experiment(self):
        print("[Start experiment]")
        total_epoch = self.args.epochs

        best_val_acc = 0.0

        for epoch in range(1, total_epoch + 1):
            print(f"Epoch {epoch}/{total_epoch}")

            self.train()
            if self.scheduler:
                self.scheduler.step()

            train_loss = self.log_dict['train_loss'][-1]
            train_acc = self.log_dict['train_acc'][-1]
            val_loss = self.log_dict['val_loss'][-1]
            val_acc = self.log_dict['val_acc'][-1]

            is_best = ''
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                is_best = ' (Best)'

            print(f"Train Loss: {train_loss:.2f} | Train Acc: {train_acc:.2f} || Val Loss: {val_loss:.2f} | Val Acc: {val_acc:.2f}{is_best}")
            print(f"=> Learning rate: {self.optimizer.state_dict()['param_groups'][0]['lr']:.6f}")
            print("-" * 60)

        # Save args and logs
        self.args.seed = torch.initial_seed()
        self.args.cuda_seed = torch.cuda.initial_seed()
        self.args.acc = np.round(self.log_dict['val_acc'][-1], 3) if 'val_acc' in self.log_dict else None

        write_json(os.path.join(self.args.save_path, "args.json"), vars(self.args))
        write_json(os.path.join(self.args.save_path, "log_dict.json"), self.log_dict)

        print("====================================Training Finished====================================")
        print(self.net)
        print_dict(vars(self.args))
