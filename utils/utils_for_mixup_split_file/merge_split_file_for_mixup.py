import os
import torch
import shutil
import argparse


def merge_file(args):

    files = os.listdir(args.root_path)
    # files.sort(key=lambda x: int(x.split(".")[0].split("_")[0]))  # os.listdir(args.root_path) 加载的会乱序

    out_file = []

    for file in files:
        if file not in ['code', args.dataset]:
            tmp_file = torch.load(os.path.join(args.root_path, file))
            print('# Length of train sample in {} is: {}'.format(str(file), len(tmp_file)))
            out_file += torch.load(os.path.join(args.root_path, file))

    if os.path.exists(os.path.join(args.root_path, '{}_train.pth'.format(args.dataset))):
        os.remove(os.path.join(args.root_path, '{}_train.pth'.format(args.dataset)))

    print('### INFO ### Length of {} train sample: {}'.format(str(args.dataset), len(out_file)))

    if not os.path.exists(os.path.join(args.root_path, args.dataset)):
        os.makedirs(os.path.join(args.root_path, args.dataset))
    torch.save(out_file, os.path.join(args.root_path, args.dataset, '{}_train.pth'.format(args.dataset)))

    # # Length of train sample in gref_train.pth is: 85474
    # # Length of train sample in gref_umd_train.pth is: 80512
    # # Length of train sample in unc+_train.pth is: 120191
    # ### INFO ### Length of mixup train sample: 286177


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    # parser.add_argument('--root_path', default="/hdd/lhxiao/refer_LAVT/ref_data_shuffled_with_mixup/mixup_with_unc+_g/prepare", type=str, required=False)
    # # Length of train sample in gref_train.pth is: 85474
    # # Length of train sample in gref_umd_train.pth is: 80512
    # # Length of train sample in unc+_train.pth is: 120191
    # ### INFO ### Length of mixup train sample: 286177

    # parser.add_argument('--root_path', default="/hdd/lhxiao/refer_LAVT/ref_data_shuffled_with_mixup/mixup_with_rec_unc_+_g/prepare", type=str, required=False)
    # 清除 gref
    # # Length of train sample in gref_umd_train.pth is: 80512
    # # Length of train sample in unc+_train.pth is: 120191
    # # Length of train sample in unc_train.pth is: 120624
    # ### INFO ### Length of mixup train sample: 321327

    # # Length of train sample in gref_train.pth is: 85474
    # # Length of train sample in gref_umd_train.pth is: 80512
    # # Length of train sample in unc+_train.pth is: 120191
    # # Length of train sample in unc_train.pth is: 120624
    # ### INFO ### Length of mixup train sample: 406801

    #########################################################
    # Length of train sample in gref_train.pth is: 85474
    # Length of train sample in gref_umd_train.pth is: 80512
    # Length of train sample in unc+_train.pth is: 120191
    # Length of train sample in unc_train.pth is: 120624
    # Length of train sample in unc_val.pth is: 10834
    # Length of train sample in unc+_val.pth is: 10758
    # Length of train sample in gref_umd_val.pth is: 4896
    ### INFO ### Length of mixup train sample: 433289

    # parser.add_argument('--root_path', default="/hdd/lhxiao/refer_LAVT/ref_data_shuffled_with_mixup/mixup_with_rec_referit/prepare", type=str, required=False)
    # Length of train sample in gref_umd_train.pth is: 80512
    # Length of train sample in unc+_train.pth is: 120191
    # Length of train sample in unc_train.pth is: 120624
    # Length of train sample in referit_train.pth is: 54127
    ### INFO ### Length of mixup train sample: 375454

    # Length of train sample in gref_train.pth is: 85474
    # Length of train sample in gref_umd_train.pth is: 80512
    # Length of train sample in unc+_train.pth is: 120191
    # Length of train sample in unc_train.pth is: 120624
    # Length of train sample in unc_val.pth is: 10834
    # Length of train sample in unc+_val.pth is: 10758
    # Length of train sample in gref_umd_val.pth is: 4896
    # Length of train sample in referit_train.pth is: 54127
    # Length of train sample in referit_val.pth is: 5842
    # Length of train sample in flickr_val.pth is: 14433
    ### INFO ### Length of mixup train sample: 507691

    # parser.add_argument('--root_path', default="/hdd/lhxiao/refer_LAVT/ref_data_shuffled_with_mixup/mixup_with_rec_referit_flickr/prepare", type=str, required=False)
    # Length of train sample in gref_train.pth is: 85474
    # Length of train sample in gref_umd_train.pth is: 80512
    # Length of train sample in unc+_train.pth is: 120191
    # Length of train sample in unc_train.pth is: 120624
    # Length of train sample in referit_train.pth is: 54127
    # Length of train sample in flickr_train.pth is: 427193
    ### INFO ### Length of mixup train sample: 888121

    # parser.add_argument('--root_path', default="/hdd/lhxiao/refer_LAVT/ref_data_shuffled_with_mixup/mixup_with_rec_referit_flickr_with_val/prepare", type=str, required=False)
    # # Length of train sample in gref_train.pth is: 85474
    # # Length of train sample in gref_umd_train.pth is: 80512
    # # Length of train sample in unc+_train.pth is: 120191
    # # Length of train sample in unc_train.pth is: 120624
    # # Length of train sample in unc_val.pth is: 10834
    # # Length of train sample in unc+_val.pth is: 10758
    # # Length of train sample in gref_umd_val.pth is: 4896
    # # Length of train sample in referit_train.pth is: 54127
    # # Length of train sample in referit_val.pth is: 5842
    # # Length of train sample in flickr_val.pth is: 14433
    # # Length of train sample in flickr_train.pth is: 427193
    # ### INFO ### Length of mixup train sample: 934884

    parser.add_argument('--root_path', default="/hdd/lhxiao/refer_LAVT/ref_data_shuffled_with_mixup/mixup_with_rec_for_all_train_val_test/prepare", type=str, required=False)
    # # Length of train sample in unc_testA.pth is: 5657
    # # Length of train sample in unc_testB.pth is: 5095
    # # Length of train sample in unc_train.pth is: 120624
    # # Length of train sample in unc_val.pth is: 10834
    # # Length of train sample in unc+_testA.pth is: 5726
    # # Length of train sample in unc+_testB.pth is: 4889
    # # Length of train sample in unc+_train.pth is: 120191
    # # Length of train sample in unc+_val.pth is: 10758
    # # Length of train sample in gref_train.pth is: 85474
    # # Length of train sample in gref_val.pth is: 9536
    # # Length of train sample in gref_umd_test.pth is: 9602
    # # Length of train sample in gref_umd_train.pth is: 80512
    # # Length of train sample in gref_umd_val.pth is: 4896
    # ### INFO ### Length of mixup train sample: 473794


    parser.add_argument('--dataset', default="mixup", type=str, required=False)
    args = parser.parse_args()

    merge_file(args)
    # get_pseudo_train_number()
