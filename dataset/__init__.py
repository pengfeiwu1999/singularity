import pdb

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from dataset.caption_dataset import (
    ImgTxtRetTrainDataset, ImgTxtRetEvalDataset,
    VidTxtRetTrainDataset, VidTxtRetEvalDataset,
    VidTxtRetMCEvalDataset
)
from dataset.qa_dataset import ImageQADataset, VideoQADataset

from dataset.dataloader import MetaLoader


def get_media_type(dataset_config):
    if len(dataset_config) == 3 and dataset_config[2] == "video":
        return "video"
    else:
        return "image"


def create_dataset(dataset_type, config):
    if config.vit_type == "deit":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif config.vit_type in ["beit", "vit"]:
        mean = (0.5, 0.5, 0.5)  # for all beit model except IN1K finetuning
        std = (0.5, 0.5, 0.5)
    else:
        raise ValueError

    normalize = transforms.Normalize(mean, std)

    # loaded images and videos are torch.Tensor of torch.uint8 format,
    # ordered as (T, 1 or 3, H, W) where T=1 for image
    type_transform = transforms.Lambda(lambda x: x.float().div(255.))

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            config.image_res, scale=(0.5, 1.0),
            interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        type_transform,
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize(
            (config.image_res, config.image_res),
            interpolation=InterpolationMode.BICUBIC),
        type_transform,
        normalize,
    ])

    video_only_dataset_kwargs_train = dict(
        video_reader_type=config.video_input.reader,
        sample_type=config.video_input.sample_type,
        num_frames=config.video_input.num_frames,
        num_tries=3,  # false tolerance
    )
    video_only_dataset_kwargs_eval = dict(
        video_reader_type=config.video_input.reader,
        sample_type=config.video_input.sample_type_test,
        num_frames=config.video_input.num_frames_test,
        num_tries=1,  # we want to have predictions for all videos
    )

    if dataset_type in ["ret_train", "ret_eval"]:  # for didemo and activitynet captions
        is_paragraph_retrieval = config.get("is_paragraph_retrieval", False)
        video_only_dataset_kwargs_eval["is_paragraph_retrieval"] = is_paragraph_retrieval
        video_only_dataset_kwargs_train["is_paragraph_retrieval"] = is_paragraph_retrieval

    if dataset_type in ["pt_train", "ret_train"]:
        # convert to list of lists
        train_files = [config.train_file] \
            if isinstance(config.train_file[0], str) else config.train_file
        train_media_types = sorted(list({get_media_type(e) for e in train_files}))
        if dataset_type == "ret_train":
            assert len(train_media_types) == 1, \
                f"retrieval downstream should only have one media type, got {train_media_types}"

        train_datasets = []
        for m in train_media_types:
            dataset_cls = ImgTxtRetTrainDataset if m == "image" else VidTxtRetTrainDataset
            # dataset of the same media_type will be mixed in a single Dataset object
            _train_files = [e for e in train_files if get_media_type(e) == m]
            dataset_kwargs = dict(
                ann_file=_train_files, transform=train_transform,
                has_multi_vision_gt=config.get("has_multi_vision_gt", False)  # true for ssv2 ret
            )
            if m == "video":
                dataset_kwargs.update(video_only_dataset_kwargs_train)
            train_datasets.append(dataset_cls(**dataset_kwargs))
        return train_datasets

    elif dataset_type in ["pt_eval", "ret_eval"]:
        test_datasets = []
        test_dataset_names = []
        # multiple test datasets, all separate
        for name, data_cfg in config.test_file.items():
            media_type = get_media_type(data_cfg)
            test_dataset_cls = ImgTxtRetEvalDataset if media_type == "image" else VidTxtRetEvalDataset
            test_dataset_names.append(name)
            dataset_kwargs = dict(
                ann_file=[data_cfg], transform=test_transform,
                has_multi_vision_gt=config.get("has_multi_vision_gt", False)  # true for ssv2 ret
            )
            if media_type == "video":
                dataset_kwargs.update(video_only_dataset_kwargs_eval)
            test_datasets.append(test_dataset_cls(**dataset_kwargs))
        return test_datasets, test_dataset_names

    elif dataset_type == "qa_train":
        media_type = get_media_type(config.train_file[0])  # assuming single train media type
        dataset_cls = ImageQADataset if media_type == "image" else VideoQADataset
        dataset_kwargs = dict(
            ann_file=config.train_file, transform=train_transform, eos=config.eos, mode="train")
        if media_type == "video":
            dataset_kwargs.update(video_only_dataset_kwargs_train)
        train_dataset = dataset_cls(**dataset_kwargs)
        return train_dataset

    elif dataset_type == "qa_eval":
        test_datasets = []
        test_dataset_names = []
        # multiple test datasets, all separate
        for name, data_cfg in config.test_file.items():
            media_type = get_media_type(data_cfg)
            test_dataset_cls = ImageQADataset if media_type == "image" else VideoQADataset
            test_dataset_names.append(name)
            dataset_kwargs = dict(
                ann_file=[data_cfg], transform=test_transform,
                eos=config.eos, mode="eval", answer_list=config.answer_list
            )
            if media_type == "video":
                dataset_kwargs.update(video_only_dataset_kwargs_eval)
            test_datasets.append(test_dataset_cls(**dataset_kwargs))
        return test_datasets, test_dataset_names

    elif dataset_type == "mc_test":
        dataset_kwargs = dict(
            ann_file=[config.test_file.mc_test], transform=test_transform
        )
        dataset_kwargs.update(video_only_dataset_kwargs_eval)
        return VidTxtRetMCEvalDataset(**dataset_kwargs)

    elif dataset_type == "star_train":
        dataset_kwargs = dict(
            ann_file=[config.train_file], transform=train_transform
        )
        dataset_kwargs.update(video_only_dataset_kwargs_train)
        return VidTxtRetMCEvalDataset(**dataset_kwargs)

    elif dataset_type == "star_test":    # 这里需要提前把测试集视频里面抽取的帧抽出来，保证结果可复现
        dataset_kwargs = dict(
            ann_file=[config.test_file.star_test], transform=test_transform
        )
        dataset_kwargs.update(video_only_dataset_kwargs_eval)
        return VidTxtRetMCEvalDataset(**dataset_kwargs)


def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list, dim=0), \
        question_list, answer_list, torch.Tensor(weight_list), n


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in \
            zip(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):

        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
            persistent_workers=True
        )
        loaders.append(loader)
    return loaders


def iterate_dataloaders(dataloaders):
    """Alternatively generate data from multiple dataloaders,
    since we use `zip` to concat multiple dataloaders,
    the loop will end when the smaller dataloader runs out.

    Args:
        dataloaders List(DataLoader): can be a single or multiple dataloaders
    """
    for data_tuples in zip(*dataloaders):
        for idx, data in enumerate(data_tuples):
            yield dataloaders[idx].dataset.media_type, data
