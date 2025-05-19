import functools, torch, os, tqdm
from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl
apply_liger_kernel_to_qwen2_vl() # important. our model is trained with this. keep consistency
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, LogitsProcessor, logging
from livecc_utils import prepare_multiturn_multimodal_inputs_for_generation, get_smart_resized_clip, get_smart_resized_video_reader
from qwen_vl_utils import process_vision_info

class LiveCCDemoInfer:
  fps = 2
  initial_fps_frames = 6
  streaming_fps_frames = 2
  initial_time_interval = initial_fps_frames / fps
  streaming_time_interval = streaming_fps_frames / fps
  frame_time_interval = 1 / fps
  def __init__(self, model_path: str = None, device_id: int = 0):
      self.model = Qwen2VLForConditionalGeneration.from_pretrained(
          model_path, torch_dtype="auto", 
          device_map=f'cuda:{device_id}', 
          attn_implementation='flash_attention_2'
      )
      self.processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
      self.model.prepare_inputs_for_generation = functools.partial(prepare_multiturn_multimodal_inputs_for_generation, self.model)
      message = {
          "role": "user",
          "content": [
              {"type": "text", "text": 'livecc'},
          ]
      }
      texts = self.processor.apply_chat_template([message], tokenize=False)
      self.system_prompt_offset = texts.index('<|im_start|>user')
      self._cached_video_readers_with_hw = {}


  def live_cc(
      self,
      query: str,
      state: dict,
      max_pixels: int = 384 * 28 * 28,
      default_query: str = 'Please describe the video.',
      do_sample: bool = True,
      repetition_penalty: float = 1.05,
      **kwargs,
  ): 
      """
      state: dict, (maybe) with keys:
          video_path: str, video path
          video_timestamp: float, current video timestamp
          last_timestamp: float, last processed video timestamp
          last_video_pts_index: int, last processed video frame index
          video_pts: np.ndarray, video pts
          last_history: list, last processed history
          past_key_values: llm past_key_values
          past_ids: past generated ids
      """
      # 1. preparation: video_reader, and last processing info
      video_timestamp, last_timestamp = state.get('video_timestamp', 0), state.get('last_timestamp', -1 / self.fps)
      video_path = state['video_path']
      if video_path not in self._cached_video_readers_with_hw:
          self._cached_video_readers_with_hw[video_path] = get_smart_resized_video_reader(video_path, max_pixels)
          video_reader = self._cached_video_readers_with_hw[video_path][0]
          video_reader.get_frame_timestamp(0)
          state['video_pts'] = torch.from_numpy(video_reader._frame_pts[:, 1])
          state['last_video_pts_index'] = -1
      video_pts = state['video_pts']
      if last_timestamp + self.frame_time_interval > video_pts[-1]:
          state['video_end'] = True
          return 
      video_reader, resized_height, resized_width = self._cached_video_readers_with_hw[video_path]
      last_video_pts_index = state['last_video_pts_index']

      # 2. which frames will be processed
      initialized = last_timestamp >= 0
      if not initialized:
          video_timestamp = max(video_timestamp, self.initial_time_interval)
      if video_timestamp <= last_timestamp + self.frame_time_interval:
          return
      timestamps = torch.arange(last_timestamp + self.frame_time_interval, video_timestamp, self.frame_time_interval) # add compensation
      
      # 3. fetch frames in required timestamps
      clip, clip_timestamps, clip_idxs = get_smart_resized_clip(video_reader, resized_height, resized_width, timestamps, video_pts, video_pts_index_from=last_video_pts_index+1)
      state['last_video_pts_index'] = clip_idxs[-1]
      state['last_timestamp'] = clip_timestamps[-1]

      # 4. organize to interleave frames
      interleave_clips, interleave_timestamps = [], []
      if not initialized:
          interleave_clips.append(clip[:self.initial_fps_frames])
          interleave_timestamps.append(clip_timestamps[:self.initial_fps_frames])
          clip = clip[self.initial_fps_frames:]
          clip_timestamps = clip_timestamps[self.initial_fps_frames:]
      if len(clip) > 0:
          interleave_clips.extend(list(clip.split(self.streaming_fps_frames)))
          interleave_timestamps.extend(list(clip_timestamps.split(self.streaming_fps_frames)))

      # 5. make conversation and send to model
      for clip, timestamps in zip(interleave_clips, interleave_timestamps):
          start_timestamp, stop_timestamp = timestamps[0].item(), timestamps[-1].item() + self.frame_time_interval
          message = {
              "role": "user",
              "content": [
                  {"type": "text", "text": f'Time={start_timestamp:.1f}-{stop_timestamp:.1f}s'},
                  {"type": "video", "video": clip}
              ]
          }
          if not query and not state.get('query', None):
              query = default_query
              print(f'No query provided, use default_query={default_query}')
          if query and state.get('query', None) != query:
              message['content'].append({"type": "text", "text": query})
              state['query'] = query
          texts = self.processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True, return_tensors='pt')
          past_ids = state.get('past_ids', None)
          if past_ids is not None:
              texts = '<|im_end|>\n' + texts[self.system_prompt_offset:]
          inputs = self.processor(
              text=texts,
              images=None,
              videos=[clip],
              return_tensors="pt",
              return_attention_mask=False
          )
          inputs.to('cuda')
          if past_ids is not None:
              inputs['input_ids'] = torch.cat([past_ids, inputs.input_ids], dim=1) 
          outputs = self.model.generate(
              **inputs, past_key_values=state.get('past_key_values', None), 
              return_dict_in_generate=True, do_sample=do_sample, 
              repetition_penalty=repetition_penalty,
          )
          state['past_key_values'] = outputs.past_key_values
          state['past_ids'] = outputs.sequences[:, :-1]
          yield (start_timestamp, stop_timestamp), self.processor.decode(outputs.sequences[0, inputs.input_ids.size(1):], skip_special_tokens=True), state

import pyrealsense2 as rs
import numpy as np
import torch
import cv2
import time
from collections import deque

# 初始化 RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# 初始化 LiveCC 模型
infer = LiveCCDemoInfer(model_path='chenjoya/LiveCC-7B-Instruct')

# 參數設置
clip_length = 6  # 每段 clip 有多少張影像
fps = 2          # 處理頻率（例如每 0.5 秒處理一次）
frame_interval = 1.0 / fps
frame_queue = deque(maxlen=clip_length)
last_time = time.time()

print("📷 開始從 RealSense 讀取影像並生成 caption...")

try:
    while True:
        # 讀取一幀
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # 轉換為 numpy array 並儲存到佇列中
        color_image = np.asanyarray(color_frame.get_data())  # BGR
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)  # To RGB
        frame_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1)  # (C, H, W)
        frame_queue.append(frame_tensor)

        # 顯示畫面
        cv2.imshow('RealSense RGB', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 每隔 frame_interval 秒產生一次 caption
        now = time.time()
        if now - last_time >= frame_interval and len(frame_queue) == clip_length:
            clip_tensor = torch.stack(list(frame_queue))  # (T, C, H, W)

            message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Live from RealSense"},
                    {"type": "video", "video": clip_tensor}
                ]
            }

            texts = infer.processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True, return_tensors='pt')
            inputs = infer.processor(
                text=texts,
                videos=[clip_tensor],
                return_tensors="pt",
                return_attention_mask=False
            ).to('cuda')

            outputs = infer.model.generate(
                **inputs,
                do_sample=True,
                repetition_penalty=1.05,
                return_dict_in_generate=True,
                # max_new_tokens=128
            )

            result = infer.processor.decode(outputs.sequences[0, inputs.input_ids.size(1):], skip_special_tokens=True)
            print("🗣️ Caption:", result)

            last_time = now

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
