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


def train(model, train_loader, optimizer, tokenizer, epoch, global_step,
          device, scheduler, scaler, config):
    model.train()
    num_options_per_q = 4
    Loss = nn.CrossEntropyLoss()
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
        label = torch.zeros(pred_prob.shape).scatter_(1, ans.unsqueeze(1), 1).to(device)
        loss = Loss(pred_prob, label)
        print("loss:", loss)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        global_step += 1

    return global_step


def main(config):
    # if is_main_process() and config.wandb.enable:
    #     run = setup_wandb(config)

    logger.info(f"config: \n{config}")
    logger.info(f"train_file: {config.train_file}")

    setup_seed(config.seed + get_rank())
    device = torch.device(config.device)
    cudnn.benchmark = True

    train_dataset = create_dataset("star_train", config)

    train_loader = create_loader(
        [train_dataset], [None],
        batch_size=[config.batch_size.video],
        num_workers=[config.num_workers],
        is_trains=[True],
        collate_fns=[None]
    )[0]

    num_steps_per_epoch = len(train_loader)
    config.scheduler.num_training_steps = num_steps_per_epoch * config.scheduler.epochs
    config.scheduler.num_warmup_steps = num_steps_per_epoch * config.scheduler.warmup_epochs

    model, model_without_ddp, optimizer, scheduler, scaler, \
    tokenizer, start_epoch, global_step = setup_model(
        config,
        model_cls=Singularity,
        has_decoder=False,
        pretrain=False,
        find_unused_parameters=True
    )
    # if is_main_process() and config.wandb.enable:
    #     wandb.watch(model)

    model = model_without_ddp
    best = 0
    best_epoch = 0
    logger.info("Start " + "evaluation" if config.evaluate else "training")
    start_time = time.time()
    for epoch in range(start_epoch, config.scheduler.epochs):
        global_step = train(
            model, train_loader, optimizer, tokenizer, epoch, global_step,
            device, scheduler, scaler, config
        )
        logger.info(f"Epoch {epoch}")
        save_obj = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "config": config,
            "epoch": epoch,
            "global_step": global_step,
        }
        model_name = "17m_keyframe1_ckpt_" + str(config.video_input.sample_type) + str(config.video_input.num_frames)+str(epoch) + ".pth"
        torch.save(save_obj, join(config.output_dir, model_name))


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")
    logger.info(f"best epoch {best_epoch} [config.stop_key {config.stop_key}]")
    logger.info(f"Checkpoints and Logs saved at {config.output_dir}")

    # if is_main_process() and config.wandb.enable:
    #     run.finish()


if __name__ == "__main__":
    cfg = setup_main()
    main(cfg)
    # if not cfg.evaluate:
    #     eval_after_training(cfg)
