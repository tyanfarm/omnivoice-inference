import base64
import binascii


def convert_audio_base64_to_file(audio_base64: str, output_path: str) -> str:
    """Decode a base64 audio string and write it to an audio file."""
    if not audio_base64:
        raise ValueError("audio_base64 must not be empty")

    payload = audio_base64.split(",", 1)[1] if "," in audio_base64 else audio_base64
    payload = "".join(payload.split())

    try:
        audio_bytes = base64.b64decode(payload, validate=True)
    except binascii.Error as exc:
        raise ValueError("audio_base64 is not valid base64 audio data") from exc

    with open(output_path, "wb") as audio_file:
        audio_file.write(audio_bytes)

    return output_path


def convert_audio_base64_text_file_to_audio_file(input_path: str, output_path: str) -> str:
    """Read base64 audio data from a text file and write the decoded audio file."""
    with open(input_path, "r", encoding="utf-8") as input_file:
        audio_base64 = input_file.read()

    return convert_audio_base64_to_file(audio_base64, output_path)


if __name__ == "__main__":
    output_file_path = "output_audio.mp3"
    convert_audio_base64_text_file_to_audio_file("file.txt", output_file_path)
    print(f"Audio file saved to: {output_file_path}")
