# -*- coding: utf-8 -*-

import logging
import torch
from torch import nn, optim

from utils import Checkpoint, Evaluator

class SupervisedTrainer:
    def __init__(self, class_dict, class_list, use_cuda):
        self.print_every = 30
        self.use_cuda = use_cuda

        self.pad_idx_in_class = class_dict['PAD_token']

        loss_weight = torch.ones(len(class_dict))
        loss_weight[self.pad_idx_in_class] = 0
        self.loss = nn.NLLLoss(weight=loss_weight, reduction="mean")
        if use_cuda:
            self.loss = self.loss.cuda()
        
        self.evaluator = Evaluator(
            class_dict=class_dict,
            class_list=class_list,
            use_cuda=use_cuda
        )
        return

    def _train_batch(self, model, inputs, input_lengths, span_length, trees, words,
                     neighbor_inputs, neighbor_input_lengths, neighbor_span_length, neighbor_trees, neighbor_words,
                     graph_map, num_pos, targets, duplicate_nums, warm_training):
        decoder_outputs, fixed_targets = model(
            inputs=inputs,
            input_lengths=input_lengths, 
            span_length=span_length,
            trees=trees,
            words=words,
            neighbor_inputs=neighbor_inputs,
            neighbor_input_lengths=neighbor_input_lengths,
            neighbor_span_length=neighbor_span_length,
            neighbor_trees=neighbor_trees,
            neighbor_words=neighbor_words,
            graph_map=graph_map,
            num_pos=num_pos,
            targets=targets,
            duplicate_nums=duplicate_nums,
            warm_training=warm_training
        )

        # replace duplicate num with max-prob token
        targets = torch.stack(fixed_targets, dim=1)

        # loss
        # batch_size * step_size * class_size
        all_decoder_outputs = torch.stack(decoder_outputs, dim=1)
        # batch_size * step_size
        mask = targets != self.pad_idx_in_class
        # valid_len * class_size, valid_len
        loss = self.loss(all_decoder_outputs[mask], targets[mask])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _train_epochs(self, data_loader, model, batch_size, start_epoch, warm_epoch, start_step, start_loss_total, max_acc, n_epoch, test_beam):
        train_list = data_loader.train_list
        test_list = data_loader.test_list

        step = start_step
        print_loss_total = start_loss_total
        max_ans_acc = max_acc

        for epoch in range(start_epoch, n_epoch + 1):
            model.train()
            warm_training = epoch <= warm_epoch
            batch_generator = data_loader.get_batch(train_list, batch_size, template_flag=True, shuffle=True)
            for batch_data_dict, batch_neighbor_dict, batch_graph_map in batch_generator:
                step += 1

                inputs = batch_data_dict['batch_span_encode_idx']
                input_lengths = batch_data_dict['batch_span_encode_len']
                span_length = batch_data_dict['batch_span_len']
                trees = batch_data_dict["batch_tree"]
                words = batch_data_dict["batch_words"]

                inputs = [torch.LongTensor(input) for input in inputs]
                input_lengths = [torch.LongTensor(input_length) for input_length in input_lengths]
                span_length = torch.LongTensor(span_length)
                if self.use_cuda:
                    inputs = [input.cuda() for input in inputs]
                    input_lengths = [input_length.cuda() for input_length in input_lengths]
                    span_length = span_length.cuda()
                
                if batch_neighbor_dict is not None and (not warm_training):
                    neighbor_inputs = batch_neighbor_dict['batch_span_encode_idx']
                    neighbor_input_lengths = batch_neighbor_dict['batch_span_encode_len']
                    neighbor_span_length = batch_neighbor_dict['batch_span_len']
                    neighbor_trees = batch_neighbor_dict["batch_tree"]
                    neighbor_words = batch_neighbor_dict["batch_words"]

                    neighbor_inputs = [torch.LongTensor(input) for input in neighbor_inputs]
                    neighbor_input_lengths = [torch.LongTensor(input_length) for input_length in neighbor_input_lengths]
                    neighbor_span_length = torch.LongTensor(neighbor_span_length)
                    if self.use_cuda:
                        neighbor_inputs = [input.cuda() for input in neighbor_inputs]
                        neighbor_input_lengths = [input_length.cuda() for input_length in neighbor_input_lengths]
                        neighbor_span_length = neighbor_span_length.cuda()
                    
                    batch_graph_map = torch.LongTensor(batch_graph_map)
                    if self.use_cuda:
                        batch_graph_map = batch_graph_map.cuda()
                else:
                    neighbor_inputs = None
                    neighbor_input_lengths = None
                    neighbor_span_length = None
                    neighbor_trees = None
                    neighbor_words = None
                
                span_num_pos = batch_data_dict["batch_span_num_pos"]
                word_num_poses = batch_data_dict["batch_word_num_poses"]
                span_num_pos = torch.LongTensor(span_num_pos)
                word_num_poses = [torch.LongTensor(word_num_pos) for word_num_pos in word_num_poses]
                if self.use_cuda:
                    span_num_pos = span_num_pos.cuda()
                    word_num_poses = [word_num_pose.cuda() for word_num_pose in word_num_poses]
                num_pos = (span_num_pos, word_num_poses)

                targets = batch_data_dict['batch_decode_idx']
                targets = torch.LongTensor(targets)
                if self.use_cuda:
                    targets = targets.cuda()
                duplicate_nums = batch_data_dict['batch_duplicate_num']
                for duplicate_num in duplicate_nums:
                    for k, vs in duplicate_num.items():
                        vs = torch.LongTensor(vs)
                        if self.use_cuda:
                            vs = vs.cuda()
                        duplicate_num[k] = vs

                loss = self._train_batch(
                    model=model,
                    inputs=inputs,
                    input_lengths=input_lengths,
                    span_length=span_length, 
                    trees=trees,
                    words=words,
                    neighbor_inputs=neighbor_inputs,
                    neighbor_input_lengths=neighbor_input_lengths,
                    neighbor_span_length=neighbor_span_length,
                    neighbor_trees=neighbor_trees,
                    neighbor_words=neighbor_words,
                    graph_map=batch_graph_map,
                    num_pos=num_pos,
                    targets=targets,
                    duplicate_nums=duplicate_nums,
                    warm_training=warm_training
                )

                print_loss_total += loss
                if step % self.print_every == 0:
                    print_loss_avg = print_loss_total / self.print_every
                    print_loss_total = 0
                    logging.info(f'step: {step}, Train loss: {print_loss_avg:.4f}')
                    # if self.use_cuda:
                    #     torch.cuda.empty_cache()
            # if self.use_cuda:
            #     torch.cuda.empty_cache()
            self.scheduler.step()

            if test_beam is None:
                template_len = True
                max_batch_size = batch_size
            else:
                template_len = False
                max_batch_size = 1
            test_temp_acc, test_ans_acc = self.evaluator.evaluate(
                model=model,
                data_loader=data_loader,
                data_list=test_list,
                template_flag=True,
                template_len=template_len,
                batch_size=max_batch_size,
                beam_width=test_beam,
                warm_training=warm_training
            )
            logging.info(f"Epoch: {epoch}, Step: {step}, test_acc: {test_temp_acc:.3f}, {test_ans_acc:.3f}")
            
            if test_ans_acc >= max_ans_acc:
                max_ans_acc = test_ans_acc
                best = True
            else:
                best = False
            Checkpoint.save(epoch=epoch, step=step, loss=print_loss_total, max_acc=max_ans_acc, model=model, optimizer=self.optimizer, scheduler=self.scheduler, best=best)
        return max_ans_acc

    def train(self, model, data_loader, batch_size, n_epoch, warm_epoch, resume,
              optim_lr, optim_weight_decay, scheduler_step_size, scheduler_gamma, test_beam):
        start_epoch = 1
        start_step = 0
        start_loss_total = 0
        max_acc = 0
        self.optimizer = optim.Adam(model.parameters(), lr=optim_lr, weight_decay=optim_weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
        if resume:
            resume_checkpoint = Checkpoint.load(model_only=False)
            model.load_state_dict(resume_checkpoint.model)
            resume_optimizer = resume_checkpoint.optimizer
            resume_scheduler = resume_checkpoint.scheduler
            if resume_optimizer is not None:
                start_epoch = resume_checkpoint.epoch + 1
                start_step = resume_checkpoint.step
                start_loss_total = resume_checkpoint.loss
                max_acc = resume_checkpoint.max_acc
                self.optimizer.load_state_dict(resume_optimizer)
                self.scheduler.load_state_dict(resume_scheduler)

        max_ans_acc = self._train_epochs(
            data_loader=data_loader, 
            model=model, 
            batch_size=batch_size,
            start_epoch=start_epoch, 
            warm_epoch=warm_epoch,
            start_step=start_step,
            start_loss_total=start_loss_total,
            max_acc=max_acc,
            n_epoch=n_epoch,
            test_beam=test_beam
        )
        return max_ans_acc
