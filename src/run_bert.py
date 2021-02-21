import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import utils
from models.bert_labeler import bert_labeler
from datasets.impressions_dataset import ImpressionsDataset
from constants import *

def collate_fn_labels(sample_list):
     """Custom collate function to pad reports in each batch to the max len
     @param sample_list (List): A list of samples. Each sample is a dictionary with
                                keys 'imp', 'label', 'len' as returned by the __getitem__
                                function of ImpressionsDataset
     
     @returns batch (dictionary): A dictionary with keys 'imp', 'label', 'len' but now
                                  'imp' is a tensor with padding and batch size as the
                                   first dimension. 'label' is a stacked tensor of labels
                                   for the whole batch with batch size as first dim. And
                                   'len' is a list of the length of each sequence in batch
     """
     tensor_list = [s['imp'] for s in sample_list]
     batched_imp = torch.nn.utils.rnn.pad_sequence(tensor_list,
                                                   batch_first=True,
                                                   padding_value=PAD_IDX)
     label_list = [s['label'] for s in sample_list]
     batched_label = torch.stack(label_list, dim=0)
     len_list = [s['len'] for s in sample_list]
     
     batch = {'imp': batched_imp, 'label': batched_label, 'len': len_list}
     return batch

def load_data(train_csv_or_logits_path, train_list_path, dev_csv_or_logits_path,
              dev_list_path, train_weights=None, batch_size=BATCH_SIZE,
              shuffle=True, num_workers=NUM_WORKERS, self_training=False):
     """ Create ImpressionsDataset objects for train and test data
     @param train_csv_or_logits_path (string): path to training csv file containing labels 
                                               or the list of logits
     @param train_list_path (string): path to list of encoded impressions for train set
     @param dev_csv_or_logits_path (string): same as train_csv_or_logits_path but for dev set
     @param dev_list_path (string): same as train_list_path but for dev set
     @param train_weights (torch.Tensor): Tensor of shape (train_set_size) containing weights
                                          for each training example, for the purposes of batch
                                          sampling with replacement
     @param batch_size (int): the batch size. As per the BERT repository, the max batch size
                              that can fit on a TITAN XP is 6 if the max sequence length
                              is 512, which is our case. We have 3 TITAN XP's
     @param shuffle (bool): Whether to shuffle data before each epoch, ignored if train_weights
                            is not None
     @param num_workers (int): How many worker processes to use to load data
     @param self_training (bool): whether to load data for self-training experiment (logits)

     @returns dataloaders (tuple): tuple of two ImpressionsDataset objects, for train and dev sets
     """
     if self_training:
        collate_fn = collate_fn_logits
        train_dset = ImpressionsLogitsDataset(train_csv_or_logits_path, train_list_path)
        dev_dset = ImpressionsLogitsDataset(dev_csv_or_logits_path, dev_list_path)
     else:
        collate_fn = collate_fn_labels
        train_dset = ImpressionsDataset(train_csv_or_logits_path, train_list_path)
        dev_dset = ImpressionsDataset(dev_csv_or_logits_path, dev_list_path)

     if train_weights is None:
          train_loader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size, shuffle=shuffle,
                                                     num_workers=num_workers, collate_fn=collate_fn)
     else:
          sampler = torch.utils.data.WeightedRandomSampler(weights=train_weights,
                                                           num_samples=len(train_weights),
                                                           replacement=True)
          train_loader = torch.utils.data.DataLoader(train_dset,
                                                     batch_size=batch_size,
                                                     num_workers=num_workers,
                                                     collate_fn=collate_fn,
                                                     sampler=sampler)
          
     dev_loader = torch.utils.data.DataLoader(dev_dset, batch_size=batch_size, shuffle=shuffle,
                                              num_workers=num_workers, collate_fn=collate_fn)
     dataloaders = (train_loader, dev_loader)
     return dataloaders

def load_test_data(test_csv_or_logits_path, test_list_path, batch_size=BATCH_SIZE, 
                   num_workers=NUM_WORKERS, shuffle=False, self_training=False):
     """ Create ImpressionsDataset object for the test set
     @param test_csv_or_logits_path (string): path to test csv file containing labels
                                              or path to list of logits 
     @param test_list_path (string): path to list of encoded impressions
     @param batch_size (int): the batch size. As per the BERT repository, the max batch size
                              that can fit on a TITAN XP is 6 if the max sequence length
                              is 512, which is our case. We have 3 TITAN XP's 
     @param num_workers (int): how many worker processes to use to load data 
     @param shuffle (bool): whether to shuffle the data or not
     @param self_training (bool): whether to load test data for self training (logits)

     @returns test_loader (dataloader): dataloader object for test set
     """
     if self_training:
        collate_fn = collate_fn_logits
        test_dset = ImpressionsLogitsDataset(test_csv_or_logits_path, test_list_path)
     else:
        collate_fn = collate_fn_labels
        test_dset = ImpressionsDataset(test_csv_or_logits_path, test_list_path)

     test_loader = torch.utils.data.DataLoader(test_dset, batch_size=batch_size, shuffle=shuffle,
                                               num_workers=num_workers, collate_fn=collate_fn)
     return test_loader

def train(save_path, dataloaders, model=None, device=None,
          optimizer=None, lr=LEARNING_RATE, log_every=LOG_EVERY,
          valid_niter=VALID_NITER, best_metric=0.0):
     """ Main training loop for the labeler
     @param save_path (string): Directory in which model weights are stored
     @param model (nn.Module): the labeler model to train, if applicable
     @param device (torch.device): device for the model. If model is not None, this
                                   parameter is required
     @param dataloaders (tuple): tuple of dataloader objects as returned by load_data
     @param optimizer (torch.optim.Optimizer): the optimizer to use, if applicable
     @param lr (float): learning rate to use in the optimizer, ignored if optimizer
                        is not None
     @param log_every (int): number of iterations to log after
     @param valid_niter (int): number of iterations after which to evaluate the model and
                               save it if it is better than old best model
     @param best_metric (float): save checkpoints only if dev set performance is higher
                                than best_metric
     """
     if model and not device:
          print("train function error: Model specified but not device")
          return
     
     if model is None:
          model = bert_labeler(pretrain_path='/data3/aihc-winter20-chexbert/bluebert/pretrain_repo')
          model.train()   #put the model into train mode
          device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
          if torch.cuda.device_count() > 1:
               print("Using", torch.cuda.device_count(), "GPUs!")
               model = nn.DataParallel(model) #to utilize multiple GPU's
          model = model.to(device)
     else:
          model.train()
          
     if optimizer is None:
          optimizer = torch.optim.Adam(model.parameters(), lr=lr)
          
     begin_time = time.time()
     report_examples = 0
     report_loss = 0.0
     train_ld = dataloaders[0]
     dev_ld = dataloaders[1]
     loss_func = nn.BCEWithLogitsLoss(reduction='sum')
     
     print('begin labeler training')
     for epoch in range(NUM_EPOCHS):
          for i, data in enumerate(train_ld, 0):
               batch = data['imp'] #(batch_size, max_len)
               batch = batch.to(device)
               label = data['label'] #(batch_size, 14)
               label = label.permute(1, 0).to(device)
               src_len = data['len']
               batch_size = batch.shape[0]
               attn_mask = utils.generate_attention_masks(batch, src_len, device)

               optimizer.zero_grad()
               out = model(batch, attn_mask) #list of 14 tensors of shape (batch_size,)
               
               batch_loss = 0.0
               for j in range(len(out)):
                    batch_loss += loss_func(out[j], label[j])
                    
               report_loss += batch_loss
               report_examples += batch_size
               loss = batch_loss / batch_size     
               loss.backward()
               optimizer.step()

               if (i+1) % log_every == 0:
                    print('epoch %d, iter %d, avg_loss %.3f, time_elapsed %.3f sec' % (epoch+1, i+1, report_loss/report_examples,
                                                                                       time.time() - begin_time))
                    report_loss = 0.0
                    report_examples = 0
                    
               if (i+1) % valid_niter == 0:
                    print('\n begin validation')
                    metrics = utils.evaluate(model, dev_ld, device)
                    auc = metrics['auc']

                    metric_avg = []
                    for j in range(len(CONDITIONS)):
                         print('%s auc: %.3f' % (CONDITIONS[j], auc[j]))
                         if CONDITIONS[j] in EVAL_CONDITIONS:
                              metric_avg.append(auc[j])
                    metric_avg = np.mean(metric_avg)
                    print('average: %.3f' % metric_avg)
                         
                    if metric_avg > best_metric: #new best network
                         print("saving new best network!\n")
                         best_metric = metric_avg
                         path = os.path.join(save_path, "model_epoch%d_iter%d" % (epoch+1, i+1))
                         torch.save({'epoch': epoch+1,
                                     'model_state_dict': model.state_dict(),
                                     'optimizer_state_dict': optimizer.state_dict()},
                                    path)

def model_from_ckpt(model, ckpt_path):
     """Load up model checkpoint
     @param model (nn.Module): the module to be loaded
     @param ckpt_path (string): path to a checkpoint. If this is None, then
                                model is trained from scratch

     @return (tuple): tuple containing the model, optimizer and device
     """
     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
     if torch.cuda.device_count() > 1:
          print("Using", torch.cuda.device_count(), "GPUs!")
          model = nn.DataParallel(model) #to utilize multiple GPU's
     model = model.to(device)
     optimizer = torch.optim.Adam(model.parameters())

     checkpoint = torch.load(ckpt_path)
     model.load_state_dict(checkpoint['model_state_dict'])
     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

     return (model, optimizer, device)

if __name__ == '__main__':
     #TRAIN ON LABELS
     #checkpoint_path = '/data3/aihc-winter20-chexbert/bluebert/mimic_reports/labeler_ckpt/auto/model_epoch6_iter8000'
     #model, optimizer, device = model_from_ckpt(bert_labeler(), checkpoint_path)
     #dataloaders = load_data('/data3/aihc-winter20-chexbert/MIMIC-CXR/vision_labeled/mimic_master_vision_labels.csv',
     #                        '/data3/aihc-winter20-chexbert/bluebert/vision_labels/impressions_lists/mimic_master',
     #                        '/data3/aihc-winter20-chexbert/chexpert_data/valid_vision_200.csv',
     #                        '/data3/aihc-winter20-chexbert/bluebert/vision_labels/impressions_lists/valid_vision_200') 
     #train(save_path='/data3/aihc-winter20-chexbert/bluebert/vision_labels/labeler_ckpt/eval_cond',
     #      dataloaders=dataloaders)


     #TEST
     model = bert_labeler()
     checkpoint_path = '/data3/aihc-winter20-chexbert/bluebert/vision_labels/labeler_ckpt/model_epoch3_iter2000'
     test_loader = load_test_data('/data3/aihc-winter20-chexbert/chexpert_data/vision_test_gt.csv',
                                  '/data3/aihc-winter20-chexbert/bluebert/vision_labels/impressions_lists/vision_test')
     utils.test(model, checkpoint_path, test_loader)

    
