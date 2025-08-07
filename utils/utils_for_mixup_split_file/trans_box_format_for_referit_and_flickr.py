import os
import torch
import shutil
import argparse


def merge_file(args):

    files = os.listdir(args.root_path)
    # files.sort(key=lambda x: int(x.split(".")[0].split("_")[0]))

    split_list = ["test", "train", "trainval", "val"]

    for split in split_list:
        if os.path.exists(os.path.join(args.root_path, '{}_{}.pth'.format(args.dataset, split))):
            file_name = '{}_{}.pth'.format(args.dataset, split)
            file = torch.load(os.path.join(os.path.join(args.root_path, file_name)))
            print('# Length of sample in {} is: {}'.format(str(file_name), len(file)))

            assert args.dataset == 'referit' or args.dataset == 'flickr'

            reformat_file = []
            for sample in file:
                if args.dataset == 'flickr':  # flickr 默认是 [img file, box, phrase]
                    bbox = [sample[1][0], sample[1][1], sample[1][2] - sample[1][0],
                            sample[1][3] - sample[1][1]]  # xyxy2xywh
                    tmp_sample = [sample[0], "flickr", bbox, sample[2], "placeholder"]
                elif args.dataset == 'referit':  # referit
                    # ('9807.jpg', '9807_13.pth', [241, 249, 313, 291], 'desk middle left of pink hat', [('r1',
                    # ['desk']), ('r2', ['none']), ('r3', ['none']), ('r4', ['left']), ('r5', ['none']), ('r6',
                    # ['none']), ('r7', ['none']), ('r8', ['pink', 'hat', 'middle'])])
                    # tmp_sample = [sample[0], sample[1], sample[2], sample[3], sample[4]]
                    bbox = [sample[2][0], sample[2][1], sample[2][2] - sample[2][0],
                            sample[2][3] - sample[2][1]]  # xyxy2xywh
                    tmp_sample = [sample[0], "referit", sample[2], sample[3], sample[4]]
                    tmp_sample[2] = bbox

                reformat_file.append(tmp_sample)

            if not os.path.exists(os.path.join(args.root_path, "box_xywh")):
                os.makedirs(os.path.join(args.root_path, "box_xywh"))
            torch.save(reformat_file, os.path.join(args.root_path, "box_xywh", file_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

    # parser.add_argument('--root_path', default="/hdd/lhxiao/refer_LAVT/ref_data_shuffled_with_mixup/referit_xywh", type=str, required=False)
    # parser.add_argument('--dataset', default="referit", type=str, required=False)
    # Length of sample in referit_test.pth is: 60103
    # Length of sample in referit_train.pth is: 54127
    # Length of sample in referit_trainval.pth is: 59969
    # Length of sample in referit_val.pth is: 5842

    parser.add_argument('--root_path', default="/hdd/lhxiao/refer_LAVT/ref_data_shuffled_with_mixup/flickr_xywh", type=str, required=False)
    parser.add_argument('--dataset', default="flickr", type=str, required=False)
    # # Length of sample in flickr_test.pth is: 14481
    # # Length of sample in flickr_train.pth is: 427193
    # # Length of sample in flickr_val.pth is: 14433

    args = parser.parse_args()

    merge_file(args)
    # get_pseudo_train_number()
