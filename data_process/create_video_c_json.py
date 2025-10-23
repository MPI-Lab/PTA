import os
import json
import random
import argparse


'''
Set your own data path: XXXXXX
'''
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--clean-path', type=str, default='/ XXXXXX/json_csv_files/ks50/clean/severity_0.json')
parser.add_argument('--video-c-path', type=str, default="/ XXXXXX/Kinetics50/image_mulframe_test-C")
parser.add_argument('--audio-path', type=str, default="/ XXXXXX/Kinetics50/audio_test")
parser.add_argument('--corruption', nargs='*', default=['all'])
args = parser.parse_args()

json_file_path = args.clean_path
with open(json_file_path, 'r') as f:
    data = json.load(f)

tmp_dic_list = data['data']

severity_list = range(1, 6)
if args.corruption[0] == 'all':
    corruption_list = [
    'gaussian_noise',
    'shot_noise',
    'impulse_noise',

    'defocus_blur',
    'glass_blur',
    'motion_blur',
    'zoom_blur',

    'snow',
    'frost',
    'fog',
    'brightness',

    'contrast',
    'elastic_transform',
    'pixelate',
    'jpeg_compression',
    ]
else:
    corruption_list = args.corruption


for corruption in corruption_list:
    mixed_severity_list = []
    for severity in severity_list:
        save_path = os.path.join(os.path.dirname(args.clean_path)[:-5], 'video')

        if not os.path.exists(os.path.join(save_path, corruption)):
            os.makedirs(os.path.join(save_path, corruption))
        dic_list = []
        for dic in tmp_dic_list:
            new_dic = {
                "video_id": dic.get("video_id"), # + '-{}-{}'.format(method, severity),
                "wav": os.path.join(args.audio_path, '{}.wav'.format(dic.get("video_id"))),
                "video_path": os.path.join(args.video_c_path, '{}/severity_{}/'.format(corruption, severity)),
                "labels": dic.get("labels")
            }
            dic_list.append(new_dic)
        print(len(dic_list))
        random.shuffle(dic_list)
        new_json = {"data": dic_list}
        with open(os.path.join(save_path, corruption, 'severity_{}.json'.format(severity)), "w") as file1:
            json.dump(new_json, file1, indent=1)
     