import mmcv

val_frame_txt = 'data/ILSVRC/ImageSets/VID_val_frames.txt'
with open(val_frame_txt, 'r') as f:
    val_frames = f.readlines()

bboxes_result_pkl = 'work_dirs/fcos_r101_caffe_fpn_gn-head_6x_vid_bs4/result_with_level_info.pkl'
bboxes = mmcv.load(bboxes_result_pkl)

all_data = []
for val_frame, bbox in zip(val_frames, bboxes):
    video_dict = {}
    frame_id = val_frame.split(' ')[-1].strip()
    video_id = val_frame.split('/')[1].split('_')[-1]
    box = []
    score = []
    level = []
    for _bbox_of_cls in bbox:
        for _bbox in _bbox_of_cls:
            box.append(_bbox[:4].tolist())
            score.append(_bbox[4].tolist())
            level.append(_bbox[5].tolist())
    video_dict['frame_id'] = frame_id
    video_dict['video_id'] = video_id
    video_dict['boxes'] = box
    video_dict['scores'] = score
    video_dict['levels'] = level
    all_data.append(video_dict)

parsed_result_pkl = 'work_dirs/fcos_r101_caffe_fpn_gn-head_6x_vid_bs4/parsed_result.pkl'
mmcv.dump(all_data, parsed_result_pkl)

