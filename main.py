from omnivoice import OmniVoice
import torch
import torchaudio

# Reset peak memory stats before starting
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Load the model
model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cuda:0",
    dtype=torch.float16
)

file_name = "am_michael"
# Generate audio
audio = model.generate(
    # text="Artificial intelligence excels at processing massive datasets, identifying hidden patterns, and executing repetitive tasks with absolute precision. It can analyze millions of lines of code in seconds, generate baseline designs, or instantly translate languages across global networks.",
    # text="Trong kỷ nguyên số hóa phát triển với tốc độ chóng mặt như hiện nay, thế giới xung quanh chúng ta đang thay đổi từng ngày, từng giờ. Những công nghệ mới ra đời không chỉ tái định hình cách chúng ta làm việc, giao tiếp mà còn thay đổi cả cách tư duy và vận hành cuộc sống.",
    text="Trong kỷ nguyên số hóa phát triển với tốc độ chóng mặt như hiện nay, thế giới xung quanh chúng ta đang thay đổi từng ngày, từng giờ. Những công nghệ mới ra đời không chỉ tái định hình cách chúng ta làm việc, giao tiếp mà còn thay đổi cả cách tư duy và vận hành cuộc sống.",
    # For voice cloning mode:
    ref_audio=f"{file_name}.mp3",
    ref_text="Human just went farther from earth than ever before. This was the mission to go back to the moon, with the goal of eventually establishing a moon colony.",
    speed=0.9, 
    num_step=16,
    postprocess_output=False,
    denoise=False
    # language="en"
    # For voice design mode (can be used if ref_audio is not provided, or to guide generation):
    # instruct="man 60 years old",
    
    # Inference steps (iterative decoding steps):
    # num_step=50,
) # audio is a list of `torch.Tensor` with shape (1, T) at 24 kHz.

torchaudio.save(f"{file_name}_clone_vi.wav", audio[0], 24000)

# Calculate and print peak VRAM
max_vram_allocated_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
max_vram_reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)

print(f"Max VRAM Allocated: {max_vram_allocated_mb:.2f} MB")
print(f"Max VRAM Reserved: {max_vram_reserved_mb:.2f} MB")
