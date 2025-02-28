import sys
import argparse
import torch
import random
import numpy as np
from infer_stage1_x_server import stage1
from infer_stage2_x_server import stage2
from infer_postprocess_server import postprocess
import uvicorn
from fastapi import FastAPI, Form,Response

global args

parser = argparse.ArgumentParser()
# Model Configuration:
parser.add_argument("--stage1_model", type=str, default="m-a-p/YuE-s1-7B-anneal-jp-kr-cot", help="The model checkpoint path or identifier for the Stage 1 model.")
parser.add_argument("--stage2_model", type=str, default="YuE-s2-1B-general", help="The model checkpoint path or identifier for the Stage 2 model.")
parser.add_argument("--tokenizer", type=str, default="mm_tokenizer_v0.2_hf/tokenizer.model", help="The model tokenizer path")
parser.add_argument("--max_new_tokens", type=int, default=3000, help="The maximum number of new tokens to generate in one pass during text generation.")
parser.add_argument("--repetition_penalty", type=float, default=1.1, help="repetition_penalty ranges from 1.0 to 2.0 (or higher in some cases). It controls the diversity and coherence of the audio tokens generated. The higher the value, the greater the discouragement of repetition. Setting value to 1.0 means no penalty.")
parser.add_argument("--run_n_segments", type=int, default=2, help="The number of segments to process during the generation.")
parser.add_argument("--stage1_use_exl2", default=True, help="Use exllamav2 to load and run stage 1 model.")
parser.add_argument("--stage2_use_exl2", default=True,help="Use exllamav2 to load and run stage 2 model.")
parser.add_argument("--stage2_batch_size", type=int, default=4, help="The non-exl2 batch size used in Stage 2 inference.")
parser.add_argument("--stage1_cache_size", type=int, default=16384, help="The cache size used in Stage 1 inference.")
parser.add_argument("--stage2_cache_size", type=int, default=32768, help="The exl2 cache size used in Stage 2 inference.")
parser.add_argument("--quantization_stage1", type=str, default="bf16", choices=["bf16", "int8", "int4", "nf4"], help="The quantization mode of the model stage 1.")
parser.add_argument("--quantization_stage2", type=str, default="int8", choices=["bf16", "int8", "int4", "nf4"], help="The quantization mode of the model stage 2.")
parser.add_argument("--sage_attention", action="store_true", help="If set, the model will use SageAttention instead of the default scaled dot product attention.")
parser.add_argument("--sdpa", action="store_true", help="If set, the model will use SageAttention instead of the default scaled dot product attention.")
parser.add_argument("--compile", action="store_true", help="If set, the model will be compiled using Torch JIT.")
parser.add_argument("--temperature", type=float, default=1.0, help="The temperature value to use during generation.")
parser.add_argument("--use_mmgp", action="store_true", help="If set, the model will use MMGP for inference.")
parser.add_argument("--stage1_cache_mode", type=str, default="FP16", help="The cache mode used in Stage 1 inference (FP16, Q8, Q6, Q4). Quantized k/v cache will save VRAM at the cost of some speed and precision.")
parser.add_argument("--stage2_cache_mode", type=str, default="FP16", help="The cache mode used in Stage 2 inference (FP16, Q8, Q6, Q4). Quantized k/v cache will save VRAM at the cost of some speed and precision.")
parser.add_argument("--stage1_no_guidance", action="store_true", help="Disable classifier-free guidance for stage 1")
# Prompt For servber
parser.add_argument("--genres", type=str)
parser.add_argument("--lyrics", type=str)
parser.add_argument("--use_audio_prompt", action="store_true", help="If set, the model will use an audio file as a prompt during generation. The audio file should be specified using --audio_prompt_path.",)
parser.add_argument("--audio_prompt_path", type=str, default="", help="The file path to an audio file to use as a reference prompt when --use_audio_prompt is enabled.")
parser.add_argument("--prompt_start_time", type=float, default=0.0, help="The start time in seconds to extract the audio prompt from the given audio file.")
parser.add_argument("--prompt_end_time", type=float, default=30.0, help="The end time in seconds to extract the audio prompt from the given audio file.")
parser.add_argument("--use_dual_tracks_prompt", action="store_true", help="If set, the model will use dual tracks as a prompt during generation. The vocal and instrumental files should be specified using --vocal_track_prompt_path and --instrumental_track_prompt_path.",)
parser.add_argument("--vocal_track_prompt_path",type=str, default="",help="The file path to a vocal track file to use as a reference prompt when --use_dual_tracks_prompt is enabled.",)
parser.add_argument("--instrumental_track_prompt_path",type=str,default="",help="The file path to an instrumental track file to use as a reference prompt when --use_dual_tracks_prompt is enabled.",)
# Output
parser.add_argument("--output_dir", type=str, default="./output", help="The directory where generated outputs will be saved.")
parser.add_argument("--keep_intermediate", action="store_true", help="If set, intermediate outputs will be saved during processing.")
parser.add_argument("--disable_offload_model", action="store_true", help="If set, the model will not be offloaded from the GPU to CPU after Stage 1 inference.")
parser.add_argument("--cuda_idx", type=int, default=0)
parser.add_argument("--seed", type=int, default=None, help="An integer value to reproduce generation.")
# Config for xcodec and upsampler
parser.add_argument("--basic_model_config", default="./xcodec_mini_infer/final_ckpt/config.yaml", help="YAML files for xcodec configurations.")
parser.add_argument("--resume_path", default="./xcodec_mini_infer/final_ckpt/ckpt_00360000.pth", help="Path to the xcodec checkpoint.")
parser.add_argument("--config_path", type=str, default="./xcodec_mini_infer/decoders/config.yaml", help="Path to Vocos config file.")
parser.add_argument("--vocal_decoder_path", type=str, default="./xcodec_mini_infer/decoders/decoder_131000.pth", help="Path to Vocos decoder weights.")
parser.add_argument("--inst_decoder_path", type=str, default="./xcodec_mini_infer/decoders/decoder_151000.pth", help="Path to Vocos decoder weights.")
parser.add_argument("-r", "--rescale", action="store_true", help="Rescale output to avoid clipping.")

app = FastAPI()

args = parser.parse_args()

def check_exit(status: int):
    if status != 0:
        sys.exit(status)

@app.post("/generate_music/")

async def generate_music(
        stage1_model: str = Form("YuE-s1-7B-anneal-jp-kr-cot"),
        stage2_model: str = Form("YuE-s2-1B-general"),
        quantization_stage1: str = Form("int8"),
        quantization_stage2: str = Form("int8"),
          stage1_use_exl2: bool = Form(True),
        stage2_use_exl2: bool = Form(True),
        genres: str = Form(...),#
        lyrics: str = Form(...),#
        max_new_tokens: int = Form(4000),#
        run_n_segments: int = Form(2),#
        repetition_penalty: float = Form(1.1),#
        stage2_batch_size: int = Form(4),#
        # Code is not using audio prompt read from local_file, but API would be updated to accept audio file 
        #audio_prompt: UploadFile = Form(None),#
        #vocals_ids: UploadFile = Form(None),
        #instrumental_ids: UploadFile = Form(None),
        use_dual_tracks_prompt: bool = Form(False),
        use_audio_prompt: bool = Form(False),
        prompt_start_time: float = Form(0.0),
        prompt_end_time: float = Form(30.0),
        top_p: float = Form(0.93),
        temperature: float = Form(1.0),
        rescale: bool = Form(False),
        seed: int = Form(42),
        ):
    global args
    args.stage1_model = stage1_model
    args.stage2_model = stage2_model
    args.quantization_stage1 = quantization_stage1
    args.quantization_stage2 = quantization_stage2

    args.stage1_use_exl2 = stage1_use_exl2
    args.stage2_use_exl2 = stage2_use_exl2
    args.genres = genres
    args.lyrics = lyrics
    args.max_new_tokens = max_new_tokens
    args.run_n_segments = run_n_segments
    args.repetition_penalty = repetition_penalty
    args.stage2_batch_size = stage2_batch_size
    # Code is not using audio prompt read from local_file, but API would be updated to accept audio file 
    #args.audio_prompt_path = audio_prompt 
    #args.vocal_track_prompt_path = vocals_ids
    #args.instrumental_track_prompt_path = instrumental_ids
    args.use_dual_tracks_prompt = use_dual_tracks_prompt
    args.use_audio_prompt = use_audio_prompt
    args.prompt_start_time = prompt_start_time
    args.prompt_end_time = prompt_end_time
    args.top_p = top_p
    args.temperature = temperature
    args.rescale = rescale
    args.seed = seed

    print("genres=",genres)
    print("lyrics=",lyrics)
    print("max_new_tokens=",max_new_tokens)
    print("run_n_segments=",run_n_segments)
    print("repetition_penalty=",repetition_penalty)
    print("stage2_batch_size=",stage2_batch_size)
    print("use_audio_prompt=",use_audio_prompt)
    print("use_dual_tracks_prompt=",use_dual_tracks_prompt)
    print("prompt_start_time=",prompt_start_time)
    print("prompt_end_time=",prompt_end_time)
    print("top_p=",top_p)
    print("temperature=",temperature)
    print("rescale=",args.rescale)
    print("seed=",args.seed)

    def seed_everything(seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    seed_everything(args.seed)

    print("Starting stage 1...")
    stage1(args)
    print("Starting stage 2...")
    stage2(args)
    print("Starting postprocessing...")
    postprocess(args)

    # track.mp3 & instrumental.mp3もnultipertで送信できるけど、フロントでの扱いが決まっていないのでmixed.mp3だけ返す
    #with opem("./output/vocoder/stems/vtrack.mp3", "rb") as f:
    #    vocal_data = f.read()
    #with open("./output/vocoder/stems/itrack.mp3", "rb") as f:
    #    instrumental_data = f.read()
    o_file="./output/mixed.mp3"
    with open(o_file, "rb") as f:
        mp3_data = f.read()
    return Response(content=mp3_data, media_type="audio/mpeg")

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
