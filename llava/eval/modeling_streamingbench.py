
import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import ast
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from typing import Dict, Optional, Sequence, List
import transformers
import re

from PIL import Image
import math

import random
import numpy as np
from glob import glob
from decord import VideoReader, cpu, bridge
bridge.set_bridge("torch")
import sys
import warnings
warnings.filterwarnings("ignore")


def split_list(lst, n):
    """
    Round-robin split:
    item0 -> chunk0, item1 -> chunk1, ..., item(n-1) -> chunk(n-1),
    itemn -> chunk0, ...
    """
    chunks = [[] for _ in range(n)]
    for i, item in enumerate(lst):
        chunks[i % n].append(item)
    return chunks


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []

    source = sources
    if roles[source[0]["from"]] != roles["human"]:
        source = source[1:]

    input_id, target = [], []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
    assert len(input_id) == len(target)
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        if has_image and sentence["value"] is not None and "<image>" in sentence["value"]:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            texts = sentence["value"].split('<image>')
            _input_id = tokenizer(role).input_ids + nl_tokens 
            for i,text in enumerate(texts):
                _input_id += tokenizer(text).input_ids 
                if i<len(texts)-1:
                    _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
            _input_id += [im_end] + nl_tokens
            assert sum([i==IMAGE_TOKEN_INDEX for i in _input_id])==num_image
        else:
            if sentence["value"] is None:
                _input_id = tokenizer(role).input_ids + nl_tokens
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id
        if role == "<|im_start|>user":
            _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
        elif role == "<|im_start|>assistant":
            _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids) + 1 : -2] + [im_end] + nl_tokens
        else:
            raise NotImplementedError
        target += _target

    input_ids.append(input_id)
    targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return input_ids


import numpy as np
from decord import VideoReader, cpu


def _parse_hms_to_seconds(time_str):
    """
    将 'HH:MM:SS' 格式转换为秒数
    例如: '00:01:22' -> 82
    """
    if not time_str:
        return None

    parts = time_str.strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"end_time 格式错误: {time_str}，应为 'HH:MM:SS'")

    h, m, s = map(int, parts)
    return h * 3600 + m * 60 + s


def load_video_baseline(video_path, keyframes, end_time,args):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    fps = vr.get_avg_fps()
    video_time = total_frame_num / fps

    # 解析截止时间
    end_time_sec = _parse_hms_to_seconds(end_time) if end_time else video_time
    print('end_time_sec',end_time_sec)
    # 防止传入时间超过视频总时长
    # end_time_sec = min(end_time_sec, video_time)

    # 对应到截止帧
    # 例如 end_time=82s，则只取 0s ~ 82s 之间的帧
    max_frame_idx = min(int(end_time_sec * fps), total_frame_num - 1)

    if max_frame_idx < 0:
        raise ValueError("end_time 对应的帧范围无效")

    if keyframes and len(keyframes) > 0:
        frame_idx = [
            int(i) for i in keyframes
            if 0 <= int(i) <= max_frame_idx
        ]
    else:
        # 只在 [0, max_frame_idx] 范围内均匀采样 32 帧
        frame_idx = np.linspace(0, max_frame_idx, 32, dtype=int).tolist()

    spare_frames = vr.get_batch(frame_idx)

    frame_time = [i / fps for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])

    return spare_frames, frame_time, end_time_sec






def load_video(video_path, keyframes, end_time, args):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=12)
    total_frame_num = len(vr)
    fps = vr.get_avg_fps()

    # 防止异常 fps
    if fps is None or fps <= 0:
        fps = 1.0

    video_time = total_frame_num / fps  # 整段视频时长，单位：秒
    print('total_frame_num, video_time', total_frame_num, video_time)

    # 解析截止时间；未传则默认整段视频
    end_time_sec = _parse_hms_to_seconds(end_time) if end_time else video_time
    # end_time_sec = min(end_time_sec, video_time)
    print('end_time_sec',end_time_sec)
    # 截取片段时长：从 0 到 end_time_sec
    clip_time = end_time_sec

    # 只允许取 0 ~ end_time_sec 之间的帧
    max_frame_idx = min(int(end_time_sec * fps), total_frame_num - 1)
    print('max_frame_idx',max_frame_idx)
    if max_frame_idx < 0:
        raise ValueError("end_time 对应的帧范围无效")

    sample_fps = 0.5   

    sample_interval = max(int(round(fps / sample_fps)), 1)
    frame_idx = list(range(0, max_frame_idx + 1, sample_interval))
    # print('frame_idx',frame_idx)
    # exit(0)
    if len(frame_idx) == 0:
        frame_idx = [0]

    if frame_idx[-1] != max_frame_idx:
        frame_idx.append(max_frame_idx)

    if len(frame_idx) == 0:
        frame_idx = [0]


    frames = vr.get_batch(frame_idx)

    frame_time = [i / fps for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])

    return frames, frame_time, clip_time


def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L10
    """
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f"{choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A. B. C. D.
            if f"{choice}." in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        # pred_index = random.choice(all_choices)
        return ''
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)  # -1 will be ignored anyway
                # start_indexes = [generated_response.index(f'({can})') for can in candidates]
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index



        


def eval_model(args):
    
    # Model
    pretrained = args.model_path
    model_name = "llava_qwen"
    device = "cuda"
    device_map = "auto"
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        pretrained,
        None,
        model_name,
    )

    model.eval()
    import pandas as pd
    # Data
    # data = json.load(open(args.gt_file, 'r'))
    df = pd.read_csv(args.gt_file)

    data = df.values.tolist()
    
    gt_questions = []
    for index, item in enumerate(data):
        # print(item)
        model.get_model().foss_cache = None
        question = item[2]
        options = item[5]
        options = ast.literal_eval(options)
        qid = item[0]
        time_stamp = item[3]
        video_id = item[0].split("_", 1)[1]
        answer_id = item[4]
        # print('answer_id',answer_id)
        answer = options[int(ord(answer_id)-ord('A'))]
        # print('answer',answer)
        # duration_group = item['duration_group']
        task = item[1]

        index2ans = {}
        for i in range(len(options)):
            idx = chr(ord("A")+i)
            ans = options[i]
            index2ans[idx] = ans
        # print(index2ans)
        # exit(0)
        gt_questions.append({
            'qid': qid, 
            'question': question, 
            'option': options, 
            'time_stamp': time_stamp,
            'video_id': video_id, 
            'answer_id': answer_id, 
            'answer': answer, 
            'index2ans': index2ans,
            'task': task,
        })
        
    questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)


    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    if args.num_chunks > 1:
        output_name = f"{args.num_chunks}_{args.chunk_idx}"
    else:
        output_name = args.output_name
    answers_file = os.path.join(args.output_dir, f"{output_name}.json")
    existing_ids = set()
    if os.path.exists(answers_file):
        with open(answers_file, "r", encoding="utf-8") as f_in:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if "id" in rec:
                        existing_ids.add(rec["id"])
                except json.JSONDecodeError:
                    continue
    ans_file = open(answers_file, "a", encoding="utf-8")
    
    for line in tqdm(questions):
        qid = line["qid"]
        if qid in existing_ids:
            continue
        answer = line["answer"]
        video_id = line["video_id"]
        answer_id = line["answer_id"]
        option = line["option"]
        time_stamp = line["time_stamp"]
        index2ans = line["index2ans"]
        task = line['task']

        question = (
        f"Question:\n{line['question']}\n"
        + "Options:\n"+option[0]+'\n'+option[1]+'\n'+option[2]+'\n'+option[3] 
    )
        option_prompt = "Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option."
        question = option_prompt + "\n" + f"{question}\nAnswer with the option's letter from the given choices directly. "
        try:
            keyframes = line['keyframes']
        except:
            keyframes = []
        sample_set = {
            "id": qid, 
            "video_id": video_id,
            "question": question, 
            "answer": answer, 
            "answer_id": answer_id, 
            "task": task,
            'keyframes': keyframes
        }

        # video_path = os.path.join(args.video_dir, f'{video_id}')
        video_path = os.path.join(args.video_dir, f'{video_id.rsplit("_", 1)[0]}','video.mp4')
        print('video_path',video_path)
        
        # exit(0)
        # Check if the video exists
        if not os.path.exists(video_path):
            print(f'Miss video {video_id}')
            continue

        if os.path.exists(video_path):
            print(video_path)
            method = os.getenv("METHOD")
            print('method',method)
            try:
                if method=='baseline':
                    video, _, _ = load_video_baseline(video_path, keyframes, time_stamp,args) # [T,C,W,H]
                else:
                    video, _, _ = load_video(video_path, keyframes, time_stamp,args) # [T,C,W,H]
            except:
                print('error')
                continue
            print('frames num', len(video))
            video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().cuda()
            # print(len(video))
            video = [video]
            
            # exit(0)
        else:
            continue
        
        qs = question
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to("cuda")

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=video,
                modalities= ["video"],
                do_sample=False,
                temperature=0,
                max_new_tokens=4096,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        print('outputs',outputs)
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        parsed_pred = parse_multi_choice_response(outputs, ["A", "B", "C", "D", "E"], index2ans)
        sample_set['acc'] = str(parsed_pred == answer_id)   
        print(sample_set['acc'])
        print(parsed_pred)
        print(answer_id)
        # exit(0)
        sample_set["pred"] = outputs
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps(sample_set)+ "\n")
        ans_file.flush()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        import gc; gc.collect()
        # exit(0)
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--model-base", type=str, default=None)

    parser.add_argument("--video-dir", type=str, default="")
    parser.add_argument("--gt-file", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--output-name", type=str, default="default")
    parser.add_argument("--question-type", type=str, default="multi_choice")
    # parser.add_argument("--for_get_frames_num", type=int, default=16)

    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--test_size", type=int, default=10000000)
    args = parser.parse_args()

    eval_model(args)