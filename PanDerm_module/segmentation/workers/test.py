import os

from lightning.pytorch import Trainer
from torch.utils.data import DataLoader

from datasets.dataset_seg import load_seg_dataset
from models import load_model


def test_worker(args):
    save_path = os.path.join(args.save_name, str(args.seed))

    for fold in range(1):
        args.fold = fold
        args.dataset_folder = os.path.join(args.dataset_path, "images")
        # args.split_path = os.path.join(args.dataset_path, args.split)
        test_dataset = load_seg_dataset(args, train=False)
        model = load_model(args)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
        )

        # load corresponding model
        best_model_path = "{}/model_best_{}.ckpt".format(save_path, fold)
        # model.load_weights(best_model_path)
        model.init_weights(best_model_path)

        trainer = Trainer(accelerator="gpu", devices=args.gpu, logger=args.logger)
        trainer.test(model, dataloaders=test_loader)
