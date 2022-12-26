import pdb
import tqdm
import pandas as pd
import time
import datetime
import logging
import wandb
import os
from os.path import join
from models.utils import tile

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
from models.model_retrieval import Singularity
from tasks.pretrain import setup_dataloaders

from utils.logger import log_dict_to_wandb, setup_wandb
from utils.config_utils import setup_main
from utils.basic_utils import MetricLogger, SmoothedValue, setup_seed, flat_list_of_lists
from utils.distributed import get_rank, is_main_process
from dataset import MetaLoader
from tasks.retrieval_utils import evaluation_wrapper
from tasks.shared_utils import setup_model
from omegaconf import OmegaConf
from dataset import create_dataset, create_loader
import copy

logger = logging.getLogger(__name__)



def eval(model, train_loader, optimizer, tokenizer, epoch, global_step,
          device, scheduler, scaler, config):
    model.eval()
    num_options_per_q = 4
    pred_all = []
    label_all = []
    pred_int = []
    label_int = []
    pred_seq = []
    label_seq = []
    pred_pre = []
    label_pre = []
    pred_fea = []
    label_fea = []
    i = 0
    for image, text, ans, ann in train_loader:
        image = image.to(device, non_blocking=True)  # bsz
        text = flat_list_of_lists(list(zip(*text)))  # List(str), len=bsz*5
        text_input = tokenizer(
            text, padding="max_length", truncation=True,
            max_length=config.max_txt_l, return_tensors="pt"
        ).to(device)  # bsz, 5, ?

        # encode text
        text_feat = model.encode_text(text_input)[0]
        # encode image
        image_feat, pooled_image_feat = model.encode_image(image)

        image_feat = tile(image_feat, 0, num_options_per_q)
        image_mask = torch.ones(
            image_feat.size()[:-1], dtype=torch.long
        ).to(device, non_blocking=True)
        # pooled_image_feat = tile(pooled_image_feat, 0, num_options_per_q)
        # cross-modal encode
        output = model.get_text_encoder()(
            encoder_embeds=text_feat,
            attention_mask=text_input.attention_mask,
            encoder_hidden_states=image_feat,
            encoder_attention_mask=image_mask,
            return_dict=True,
            mode="fusion"
        )
        itm_embeds = output.last_hidden_state[:, 0]  # [CLS]

        score = model.itm_head(itm_embeds)[:, 1]
        pred_prob = score.view(-1, num_options_per_q)
        pred = pred_prob.max(1)[1].tolist()
        assert len(pred) == len(ans)
        for iter in range(len(ans)):
            if ann["question_id"][iter][:3].lower() == "int":
                pred_int.append(pred[iter])
                label_int.append(ans[iter])
                continue
            elif ann["question_id"][iter][:3].lower() == "seq":
                pred_seq.append(pred[iter])
                label_seq.append(ans[iter])
                continue
            elif ann["question_id"][iter][:3].lower() == "pre":
                pred_pre.append(pred[iter])
                label_pre.append(ans[iter])
                continue
            elif ann["question_id"][iter][:3].lower() == "fea":
                pred_fea.append(pred[iter])
                label_fea.append(ans[iter])
                continue
            else:
                print("存在第五类问题？")
                raise Exception
        pred_all.extend(pred)
        label_all.extend(ans)
        global_step += 1

    assert len(pred_all) == len(label_all)
    assert len(pred_int) == len(label_int)
    assert len(pred_seq) == len(label_seq)
    assert len(pred_pre) == len(label_pre)
    assert len(pred_fea) == len(label_fea)
    correct = 0
    for iter in range(len(pred_all)):
        if pred_all[iter] == label_all[iter]:
            correct += 1
    print("该setting下总正确率为： ", correct/len(pred_all), "总的问题的数目为:  ", len(pred_all))
    correct = 0
    for iter in range(len(pred_int)):
        if pred_int[iter] == label_int[iter]:
            correct += 1
    print("该setting下int类问题正确率为： ", correct / len(pred_int), "该类问题的数目为:  ", len(pred_int))
    correct = 0
    for iter in range(len(pred_seq)):
        if pred_seq[iter] == label_seq[iter]:
            correct += 1
    print("该setting下seq类问题正确率为： ", correct / len(pred_seq), "该类问题的数目为:  ", len(pred_seq))
    correct = 0
    for iter in range(len(pred_pre)):
        if pred_pre[iter] == label_pre[iter]:
            correct += 1
    print("该setting下pre类问题正确率为： ", correct / len(pred_pre), "该类问题的数目为:  ", len(pred_pre))
    correct = 0
    for iter in range(len(pred_fea)):
        if pred_fea[iter] == label_fea[iter]:
            correct += 1
    print("该setting下fea类问题正确率为： ", correct / len(pred_fea), "该类问题的数目为:  ", len(pred_fea))



    return global_step


def main(config):
    # if is_main_process() and config.wandb.enable:
    #     run = setup_wandb(config)

    logger.info(f"config: \n{config}")
    logger.info(f"test_file: {config.test_file}")

    setup_seed(config.seed + get_rank())
    device = torch.device(config.device)
    cudnn.benchmark = True

    # train_dataset = create_dataset("star_test", config)
    #
    # config.scheduler.num_training_steps = 10
    # config.scheduler.num_warmup_steps = 10

    for test_sample_num in [5, 10]:
        if test_sample_num == 5:
            config.batch_size_test.video = 4
        else:
            config.batch_size_test.video = 2
        train_dataset = create_dataset("star_test", config)
        config.scheduler.num_training_steps = 10
        config.scheduler.num_warmup_steps = 10

        for setting in ["uniform_uniform", "uniform_key", "key_key"]:
            for epoch_train in range(5):
                # if epoch_train != 4:
                #     continue
                if setting == "uniform_uniform":
                    # continue
                    config.pretrained_path = "/share/wupf/singularity/to/ckpts_and_logs/mc_star/star_star_uniform1/17m_uniform1_ckpt_"+str(epoch_train)+".pth"
                    config.video_input.sample_type_test = "rand"
                    train_dataset.sample_type = "rand"
                elif setting == "uniform_key":
                    # continue
                    config.pretrained_path = "/share/wupf/singularity/to/ckpts_and_logs/mc_star/star_star_uniform1/17m_uniform1_ckpt_"+str(epoch_train)+".pth"
                    config.video_input.sample_type_test = "key_frame"
                    train_dataset.sample_type = "key_frame"
                elif setting == "key_key":
                    config.pretrained_path = "/share/wupf/singularity/to/ckpts_and_logs/mc_star/star_star_key1/17m_keyframe1_ckpt_" + str(epoch_train) + ".pth"
                    config.video_input.sample_type_test = "key_frame"
                    train_dataset.sample_type = "key_frame"

                config.video_input.num_frames_test = test_sample_num

                train_loader = create_loader(
                    [train_dataset], [None],
                    batch_size=[config.batch_size_test.video],
                    num_workers=[config.num_workers],
                    is_trains=[False],
                    collate_fns=[None]
                )[0]

                model, model_without_ddp, optimizer, scheduler, scaler, \
                tokenizer, start_epoch, global_step = setup_model(
                    config,
                    model_cls=Singularity,
                    has_decoder=False,
                    pretrain=False,
                    find_unused_parameters=True
                )

                model = model_without_ddp

                print("现在测试的setting是：  " + setting + "   test_sample_num:" + str(test_sample_num) + "   epoch_train:" + str(epoch_train))
                logger.info("Start " + "evaluation")
                epoch=0
                eval(model, train_loader, optimizer, tokenizer, epoch, global_step, device, scheduler, scaler, config)


if __name__ == "__main__":
    cfg = setup_main()
    main(cfg)
    # if not cfg.evaluate:
    #     eval_after_training(cfg)
