import gradio as gr
import requests
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import uuid

# **FastAPI サーバーの URL**
API_URL = "http://localhost:8000/generate_music/"

# **top_200_tags.jsonファイルの読み込み**
with open("../YuE/top_200_tags.json", "r", encoding="utf-8") as f:
    top_200_tags = json.load(f)
genre = top_200_tags["genre"]
instrument = top_200_tags["instrument"]
mood = top_200_tags["mood"]
gender = top_200_tags["gender"]
timbre = top_200_tags["timbre"]

def generate_music(genres, lyrics, max_new_tokens, run_n_segments, repetition_penalty, stage2_batch_size, prompt_end_time, top_p, temperature, seed, stage1_model, stage2_model, language):
    quantization_stage1 = "int8" if stage1_model == "exllamav2" else "bf16"
    quantization_stage2 = "int8" if stage2_model == "exllamav2" else "bf16"

    if language == "Japanese":
        if stage1_model == "exllamav2":
            stage1_model="YuE-s1-7B-anneal-jp-kr-cot"
            stage1_use_exl2 = True
        else:
            stage1_model="m-a-p/YuE-s1-7B-anneal-jp-kr-cot"
            stage1_use_exl2 = False
    else:
        if stage1_model == "exllamav2":
            stage1_model="YuE-s1-7B-anneal-en-cot"
            stage1_use_exl2 = True
        else:
            stage1_model="m-a-p/YuE-s1-7B-anneal-en-cot"
            stage1_use_exl2 = False
    if stage2_model == "exllamav2":
        stage2_model="YuE-s2-1B-general"
        stage2_use_exl2 = True
    else:
        stage2_model="m-a-p/YuE-s2-1B-general"
        stage2_use_exl2 = False

    payload = {
        "genres": genres,  # 必須
        "lyrics": lyrics,  # 必須
        "max_new_tokens": max_new_tokens,
        "run_n_segments": run_n_segments,
        "repetition_penalty": repetition_penalty,
        "stage2_batch_size": stage2_batch_size,
        "prompt_end_time": prompt_end_time,
        "top_p": top_p,
        "temperature": temperature,
        "seed": seed,
        "stage1_model": stage1_model,
        "stage2_model": stage2_model,
        "quantization_stage1": quantization_stage1,
        "quantization_stage2": quantization_stage2,
        "stage1_use_exl2": stage1_use_exl2,
        "stage2_use_exl2": stage2_use_exl2,
        "language": language,
        "top_200_tags": top_200_tags  # 追加
    }

    response = requests.post(API_URL, data=payload)
    mp3_filename = "generated_music.mp3"

    if response.status_code == 200:
        with open(mp3_filename, "wb") as f:
            f.write(response.content)
        print(f"✅ 音楽生成成功！MP3 ファイルを保存しました: {mp3_filename}")
        return mp3_filename
    else:
        print(f"❌ エラー: {response.status_code}, メッセージ: {response.text}")
        return None

def plot_waveform(mp3_filename):
    temp_wav_filename = f"{uuid.uuid4()}.wav"
    # MP3ファイルをWAVに変換して読み込む
    os.system(f"ffmpeg -i {mp3_filename} {temp_wav_filename}")
    samplerate, data = wavfile.read(temp_wav_filename)
    times = np.arange(len(data)) / float(samplerate)

    plt.figure(figsize=(15, 5))
    plt.fill_between(times, data, color='k')
    plt.xlim(times[0], times[-1])
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Waveform of Generated Music')
    plt.savefig("waveform.png")
    print("Waveform image saved as 'waveform.png'")
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.fill_between(times, data, color='k')
    ax.set_xlim(times[0], times[-1])
    ax.set_xlabel('time (s)')
    ax.set_ylabel('amplitude')
    os.remove(temp_wav_filename)
    return fig

def main(genres, lyrics, max_new_tokens, run_n_segments, repetition_penalty, stage2_batch_size, prompt_end_time, top_p, temperature, seed, stage1_model, stage2_model, language):
    mp3_filename = generate_music(genres, lyrics, max_new_tokens, run_n_segments, repetition_penalty, stage2_batch_size, prompt_end_time, top_p, temperature, seed, stage1_model, stage2_model, language)
    if mp3_filename:
        waveform_image = plot_waveform(mp3_filename)
        return mp3_filename, waveform_image
    else:
        return "Failed to generate music."

def read_file(file):
    if file is not None:
        with open(file.name, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def update_textbox(file):
    return read_file(file)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=2):
            genres_textbox = gr.Textbox(label="ジャンル", value="inspiring female uplifting pop airy vocal electronic bright vocal vocal")
            genres_file = gr.File(label="ジャンルファイル", file_count="single", type="filepath")
            lyrics_textbox = gr.Textbox(label="歌詞", value="ここに歌詞を入力してください")
            lyrics_file = gr.File(label="歌詞ファイル", file_count="single", type="filepath")
        with gr.Column(scale=1):
            max_new_tokens = gr.Slider(minimum=101, maximum=5000, value=4000, label="Max New Tokens", step=1)
            run_n_segments = gr.Slider(minimum=1, maximum=10, value=2, label="Run N Segments", step=1)
            repetition_penalty = gr.Slider(minimum=0.1, maximum=2.0, value=1.1, label="Repetition Penalty", step=0.1)
            stage2_batch_size = gr.Slider(minimum=1, maximum=10, value=4, label="Stage 2 Batch Size", step=1)
            prompt_end_time = gr.Slider(minimum=1.0, maximum=60.0, value=40.0, label="Prompt End Time", step=0.1)
            top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.93, label="Top P", step=0.01)
            temperature = gr.Slider(minimum=0.0, maximum=2.0, value=1.0, label="Temperature", step=0.1)
            seed = gr.Number(label="Seed", value=42)
            stage1_model = gr.Radio(choices=["Original", "exllamav2"], label="Stage 1 Model", value="Original")
            stage2_model = gr.Radio(choices=["Original", "exllamav2"], label="Stage 2 Model", value="exllamav2")
            language = gr.Radio(choices=["English", "Japanese"], label="Language", value="Japanese")
            submit_btn = gr.Button("生成")
        with gr.Column(scale=2):
            output_audio = gr.Audio(type="filepath", label="生成された音楽", autoplay=True)
            output_waveform = gr.Plot(label="波形")

    def submit(genres_file, lyrics_file, genres, lyrics, max_new_tokens, run_n_segments, repetition_penalty, stage2_batch_size, prompt_end_time, top_p, temperature, seed, stage1_model, stage2_model, language):
        if genres_file:
            genres = read_file(genres_file)
        if lyrics_file:
            lyrics = read_file(lyrics_file)
        return main(genres, lyrics, max_new_tokens, run_n_segments, repetition_penalty, stage2_batch_size, prompt_end_time, top_p, temperature, seed, stage1_model, stage2_model, language)

    genres_file.change(fn=update_textbox, inputs=genres_file, outputs=genres_textbox)
    lyrics_file.change(fn=update_textbox, inputs=lyrics_file, outputs=lyrics_textbox)
    submit_btn.click(submit, inputs=[genres_file, lyrics_file, genres_textbox, lyrics_textbox, max_new_tokens, run_n_segments, repetition_penalty, stage2_batch_size, prompt_end_time, top_p, temperature, seed, stage1_model, stage2_model, language], outputs=[output_audio, output_waveform])

demo.launch()
