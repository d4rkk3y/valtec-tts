#!/usr/bin/env python3


"""
Vietnamese TTS Inference Script
Synthesizes speech from text using trained model.
"""

import os
import sys
import json
import argparse
import glob
import re
from pathlib import Path

import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm

# Local imports
from src.vietnamese.text_processor import process_vietnamese_text
from src.vietnamese.phonemizer import text_to_phonemes, VIPHONEME_AVAILABLE
from src.models.synthesizer import SynthesizerTrn
from src.text.symbols import symbols
from src.utils import helpers as utils

def find_latest_checkpoint(model_dir, prefix="G"):
    """Find the latest checkpoint in model directory."""
    pattern = os.path.join(model_dir, f"{prefix}*.pth")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None
    
    def get_step(path):
        match = re.search(rf'{prefix}(\d+)\.pth', path)
        return int(match.group(1)) if match else 0
    
    checkpoints.sort(key=get_step, reverse=True)
    return checkpoints[0]


def parse_args():
    parser = argparse.ArgumentParser(description="Vietnamese TTS Inference")
    parser.add_argument("--checkpoint", "-c", type=str, default=None,
                        help="Path to generator checkpoint (G*.pth). If not specified, uses latest from --model_dir")
    parser.add_argument("--model_dir", type=str, default="./pretrained",
                        help="Model directory to find latest checkpoint (default: ./logs/vietnamese_10ch_finetune_viphoneme)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.json (auto-detect if not specified)")
    parser.add_argument("--text", "-t", type=str, default=None,
                        help="Text to synthesize")
    parser.add_argument("--speaker", "-s", type=str, default=None,
                        help="Speaker name (from config)")
    parser.add_argument("--output", "-o", type=str, default="output.wav",
                        help="Output audio file path")
    parser.add_argument("--input_file", type=str, default=None,
                        help="Input file with texts (one per line)")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory for batch mode")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive mode")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--sdp_ratio", type=float, default=0.0,
                        help="SDP ratio (0.0 = deterministic, 1.0 = stochastic)")
    parser.add_argument("--noise_scale", type=float, default=0.667,
                        help="Noise scale for generation")
    parser.add_argument("--noise_scale_w", type=float, default=0.8,
                        help="Noise scale for duration")
    parser.add_argument("--length_scale", type=float, default=1.0,
                        help="Length scale (speed)")
    parser.add_argument("--chunk_mode", type=str, default="auto",
                        help="Chunking mode: 'auto' (default), 'sentence', 'length', or 'none'")
    parser.add_argument("--max_chunk_chars", type=int, default=200,
                        help="Maximum characters per chunk when using length mode")
    return parser.parse_args()


class VietnameseTTS:
    """Vietnamese TTS synthesizer using trained VITS-based model."""
    
    def __init__(self, checkpoint_path, config_path, device="cuda"):
        self.device = device
        
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.sampling_rate = self.config['data']['sampling_rate']
        self.spk2id = self.config['data']['spk2id']
        self.speakers = list(self.spk2id.keys())
        self.add_blank = self.config['data'].get('add_blank', True)
        
        print(f"Available speakers: {self.speakers}")
        
        # Load model
        self._load_model(checkpoint_path)
    
    def _load_model(self, checkpoint_path):
        """Load the trained model."""
        
        # Create model
        hps_data = utils.HParams(**self.config['data'])
        hps_model = utils.HParams(**self.config['model'])
        
        self.model = SynthesizerTrn(
            len(symbols),
            self.config['data']['filter_length'] // 2 + 1,
            self.config['train']['segment_size'] // self.config['data']['hop_length'],
            n_speakers=self.config['data']['n_speakers'],
            **self.config['model'],
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle DDP checkpoint
        state_dict = checkpoint['model']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.eval()
        
        print(f"Model loaded from {checkpoint_path}")
    
    def _split_text_into_chunks(self, text, mode="auto", max_chars=200):
        """
        Split text into chunks for processing.
        
        Args:
            text: Input text to split
            mode: 'auto' (sentences), 'sentence' (by punctuation), 'length' (by character count), 'none' (no splitting)
            max_chars: Maximum characters per chunk for length mode
        
        Returns:
            List of text chunks
        """
        if mode == "none":
            return [text]
        
        if mode == "auto":
            # Try sentence splitting first
            chunks = self._split_by_sentences(text, max_chars)
            if len(chunks) == 1 and len(chunks[0]) > max_chars:
                # If only one chunk and it's too long, split by length
                chunks = self._split_by_length(text, max_chars)
            return chunks
        
        elif mode == "sentence":
            return self._split_by_sentences(text, max_chars)
        
        elif mode == "length":
            return self._split_by_length(text, max_chars)
        
        else:
            return [text]
    
    def _split_by_sentences(self, text, max_chars=200):
        """
        Split text by sentence-ending punctuation, but also respect max_chars limit.
        If any sentence exceeds max_chars, it will be further split by length.
        """
        # Vietnamese sentence endings: . ! ? ... 
        # Also handle cases where punctuation might be followed by quotes or spaces
        import re
        
        # Split by punctuation while preserving the punctuation
        pattern = r'([.!?…]+[\s\'"\)\]]*|\n+)'
        parts = re.split(pattern, text)
        
        sentences = []
        current_sentence = ""
        
        for part in parts:
            if not part:
                continue
                
            # If part ends with punctuation, it's a sentence end
            if re.search(r'[.!?…]+', part):
                current_sentence += part
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ""
            else:
                # Add to current sentence
                current_sentence += part
        
        # Add remaining text
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        # If we got too many small sentences, merge them
        if len(sentences) > 1:
            merged = []
            current = ""
            for sentence in sentences:
                if len(current) + len(sentence) < 150:
                    current += " " + sentence if current else sentence
                else:
                    if current:
                        merged.append(current)
                    current = sentence
            if current:
                merged.append(current)
            sentences = merged
        
        # Fallback to original text if splitting failed
        if not sentences:
            sentences = [text]
        
        # Now check each sentence against max_chars and split long ones
        final_chunks = []
        for sentence in sentences:
            if len(sentence) <= max_chars:
                final_chunks.append(sentence)
            else:
                # This sentence is too long, split it by length
                long_chunks = self._split_by_length(sentence, max_chars)
                final_chunks.extend(long_chunks)
        
        return final_chunks
    
    def _split_by_length(self, text, max_chars):
        """
        Split text by max characters first, but avoid cutting sentences in the middle.
        This is the primary splitting method for OOM prevention.
        """
        import re
        
        # First, find all sentence boundaries
        sentence_pattern = r'([.!?…]+[\s\'"\)\]]*|\n+)'
        sentence_parts = re.split(sentence_pattern, text)
        
        # Reconstruct sentences with their boundaries
        sentences = []
        current_sentence = ""
        
        for part in sentence_parts:
            if not part:
                continue
                
            if re.search(r'[.!?…]+', part):
                current_sentence += part
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ""
            else:
                current_sentence += part
        
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        # If no clear sentences, fall back to word-based splitting
        if len(sentences) == 1 and len(sentences[0]) == len(text):
            return self._split_by_words(text, max_chars)
        
        # Now build chunks by combining sentences without exceeding max_chars
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed limit
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(test_chunk) <= max_chars:
                # It fits, add it
                current_chunk = test_chunk
            else:
                # It doesn't fit
                if current_chunk:
                    # Save current chunk
                    chunks.append(current_chunk)
                    # Start new chunk with current sentence
                    current_chunk = sentence
                else:
                    # Single sentence is too long, split it by words
                    word_chunks = self._split_by_words(sentence, max_chars)
                    chunks.extend(word_chunks)
                    current_chunk = ""
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_by_words(self, text, max_chars):
        """
        Fallback method: split by words when sentences are too long.
        Tries to avoid cutting words in half.
        """
        words = text.split()
        chunks = []
        current_chunk = ""
        
        for word in words:
            # Check if adding this word would exceed limit
            test_chunk = current_chunk + " " + word if current_chunk else word
            if len(test_chunk) <= max_chars:
                current_chunk = test_chunk
            else:
                # Save current chunk
                if current_chunk:
                    chunks.append(current_chunk)
                # Start new chunk with current word
                current_chunk = word
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        # If any chunk is still too long, split it roughly
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= max_chars:
                final_chunks.append(chunk)
            else:
                # Split roughly at character limit
                for i in range(0, len(chunk), max_chars):
                    final_chunks.append(chunk[i:i+max_chars])
        
        return final_chunks
    
    def _concatenate_audio(self, audio_chunks, sr, pause_duration=0.1):
        """
        Concatenate audio chunks with short pauses between them.
        
        Args:
            audio_chunks: List of numpy audio arrays
            sr: Sample rate
            pause_duration: Duration of pause in seconds between chunks
        
        Returns:
            Concatenated audio array
        """
        if len(audio_chunks) == 1:
            return audio_chunks[0]
        
        # Create pause audio (silence)
        pause_samples = int(pause_duration * sr)
        pause_audio = np.zeros(pause_samples)
        
        # Concatenate with pauses
        result = []
        for i, chunk in enumerate(audio_chunks):
            result.append(chunk)
            # Add pause after each chunk except the last
            if i < len(audio_chunks) - 1:
                result.append(pause_audio)
        
        return np.concatenate(result)
    
    def text_to_sequence(self, text, speaker):
        """Convert text to model input tensors."""
        from src.text import cleaned_text_to_sequence
        from src.nn import commons
        
        # Normalize text
        normalized_text = process_vietnamese_text(text)
        
        # Convert to phonemes
        phones, tones, word2ph = text_to_phonemes(normalized_text, use_viphoneme=VIPHONEME_AVAILABLE)
        
        # Convert to sequence
        phone_ids, tone_ids, lang_ids = cleaned_text_to_sequence(phones, tones, "VI")
        
        # Add blanks if needed
        if self.add_blank:
            phone_ids = commons.intersperse(phone_ids, 0)
            tone_ids = commons.intersperse(tone_ids, 0)
            lang_ids = commons.intersperse(lang_ids, 0)
        
        # Get speaker ID
        if speaker not in self.spk2id:
            print(f"Warning: Speaker '{speaker}' not found, using first speaker: {self.speakers[0]}")
            speaker = self.speakers[0]
        speaker_id = self.spk2id[speaker]
        
        # Create tensors
        x = torch.LongTensor(phone_ids).unsqueeze(0).to(self.device)
        x_lengths = torch.LongTensor([len(phone_ids)]).to(self.device)
        tone = torch.LongTensor(tone_ids).unsqueeze(0).to(self.device)
        language = torch.LongTensor(lang_ids).unsqueeze(0).to(self.device)
        sid = torch.LongTensor([speaker_id]).to(self.device)
        
        # Create dummy BERT features (zeros if disabled)
        bert = torch.zeros(1024, len(phone_ids)).unsqueeze(0).to(self.device)
        ja_bert = torch.zeros(768, len(phone_ids)).unsqueeze(0).to(self.device)
        
        return x, x_lengths, tone, language, sid, bert, ja_bert
    
    @torch.no_grad()
    def synthesize(self, text, speaker, sdp_ratio=0.0, noise_scale=0.667, 
                   noise_scale_w=0.8, length_scale=1.0, chunk_mode="auto", max_chunk_chars=200):
        """
        Synthesize speech from text with optional chunking for long texts.
        
        Args:
            text: Input Vietnamese text
            speaker: Speaker name
            sdp_ratio: Stochastic duration predictor ratio (0=deterministic)
            noise_scale: Noise scale for generation
            noise_scale_w: Noise scale for duration
            length_scale: Speed control (1.0=normal, <1.0=faster, >1.0=slower)
            chunk_mode: 'auto', 'sentence', 'length', or 'none'
            max_chunk_chars: Maximum characters per chunk
        
        Returns:
            audio: numpy array of audio samples
            sr: sample rate
        """
        # Check if text is too long and needs chunking
        if chunk_mode != "none" and len(text) > max_chunk_chars:
            print(f"Text is long ({len(text)} chars), splitting into chunks using mode: {chunk_mode}")
            chunks = self._split_text_into_chunks(text, chunk_mode, max_chunk_chars)
            
            if len(chunks) == 1:
                # Single chunk, process normally
                return self._synthesize_single_chunk(text, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale)
            
            # Process each chunk
            audio_chunks = []
            for i, chunk in enumerate(chunks):
                print(f"Processing chunk {i+1}/{len(chunks)}: {chunk[:50]}...")
                audio, sr = self._synthesize_single_chunk(chunk, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale)
                audio_chunks.append(audio)
            
            # Concatenate with pauses
            final_audio = self._concatenate_audio(audio_chunks, sr)
            return final_audio, sr
        else:
            # No chunking needed
            return self._synthesize_single_chunk(text, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale)
    
    def _synthesize_single_chunk(self, text, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale):
        """Synthesize a single chunk of text."""
        # Prepare inputs
        x, x_lengths, tone, language, sid, bert, ja_bert = self.text_to_sequence(text, speaker)
        
        # Generate
        audio, attn, *_ = self.model.infer(
            x, x_lengths, sid, tone, language, bert, ja_bert,
            sdp_ratio=sdp_ratio,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            length_scale=length_scale,
        )
        
        audio = audio[0, 0].cpu().numpy()
        
        return audio, self.sampling_rate
    
    def save_audio(self, audio, sr, output_path):
        """Save audio to file."""
        sf.write(output_path, audio, sr)
        print(f"Audio saved to {output_path}")


def _extract_iter_from_checkpoint(checkpoint_path: str) -> str | None:
    base = os.path.basename(checkpoint_path)
    m = re.search(r"G_(\d+)\.pth$", base)
    if m:
        return m.group(1)
    return None


def _append_suffix_before_ext(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def _resolve_output_path(output: str, output_dir: str, suffix: str) -> Path:
    p = Path(output)
    if not p.is_absolute():
        p = Path(output_dir) / p.name
    p = _append_suffix_before_ext(p, suffix)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def main():
    args = parse_args()
    
    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    
    # Find checkpoint if not specified
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint(args.model_dir, "G")
        if checkpoint_path is None:
            # Try to download from Hugging Face
            print(f"No checkpoint found in {args.model_dir}")
            print("Attempting to download from Hugging Face...")
            try:
                from huggingface_hub import snapshot_download
                
                # Default HF repo
                hf_repo = "valtecAI-team/valtec-tts-pretrained"
                
                # Get cache directory
                if os.name == 'nt':  # Windows
                    cache_base = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData' / 'Local'))
                else:  # Linux/Mac
                    cache_base = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))
                
                model_dir = cache_base / 'valtec_tts' / 'models' / 'vits-vietnamese'
                model_dir.mkdir(parents=True, exist_ok=True)
                
                print(f"Downloading model to: {model_dir}")
                snapshot_download(repo_id=hf_repo, local_dir=str(model_dir))
                print("Download complete!")
                
                # Update model_dir and find checkpoint
                args.model_dir = str(model_dir)
                checkpoint_path = find_latest_checkpoint(args.model_dir, "G")
                
            except Exception as e:
                print(f"Error downloading model: {e}")
                print("Please specify --checkpoint or --model_dir")
                return
            
            if checkpoint_path is None:
                print("Error: Could not find checkpoint after download")
                return
                
        print(f"Using latest checkpoint: {checkpoint_path}")

    iter_str = _extract_iter_from_checkpoint(checkpoint_path)
    iter_suffix = f"iter{iter_str}" if iter_str is not None else "iterunknown"
    
    # Auto-find config if in same directory as checkpoint
    config_path = args.config
    if config_path is None:
        config_dir = os.path.dirname(checkpoint_path)
        config_path = os.path.join(config_dir, "config.json")
        if not os.path.exists(config_path):
            print(f"Error: config.json not found at {config_path}")
            return
        print(f"Using config: {config_path}")
    
    # Initialize TTS
    print("Loading model...")
    tts = VietnameseTTS(checkpoint_path, config_path, args.device)
    
    # Get default speaker
    default_speaker = args.speaker or tts.speakers[0]
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.interactive:
        # Interactive mode
        print("\n" + "=" * 60)
        print("Vietnamese TTS - Interactive Mode")
        print(f"Default speaker: {default_speaker}")
        print(f"Available speakers: {', '.join(tts.speakers)}")
        print("Commands: 'quit' to exit, 'speaker NAME' to change speaker")
        print("          'chunk MODE' to change chunking mode (auto/sentence/length/none)")
        print("=" * 60 + "\n")
        
        current_speaker = default_speaker
        current_chunk_mode = args.chunk_mode
        
        while True:
            try:
                text = input("Enter text: ").strip()
                
                if not text:
                    continue
                
                if text.lower() == 'quit':
                    break
                
                if text.lower().startswith('speaker '):
                    new_speaker = text[8:].strip()
                    if new_speaker in tts.speakers:
                        current_speaker = new_speaker
                        print(f"Speaker changed to: {current_speaker}")
                    else:
                        print(f"Speaker not found. Available: {', '.join(tts.speakers)}")
                    continue
                
                if text.lower().startswith('chunk '):
                    new_mode = text[6:].strip().lower()
                    if new_mode in ['auto', 'sentence', 'length', 'none']:
                        current_chunk_mode = new_mode
                        print(f"Chunk mode changed to: {current_chunk_mode}")
                    else:
                        print(f"Invalid mode. Use: auto, sentence, length, or none")
                    continue
                
                # Synthesize
                print(f"Synthesizing with speaker '{current_speaker}' (chunk mode: {current_chunk_mode})...")
                audio, sr = tts.synthesize(
                    text, current_speaker,
                    sdp_ratio=args.sdp_ratio,
                    noise_scale=args.noise_scale,
                    noise_scale_w=args.noise_scale_w,
                    length_scale=args.length_scale,
                    chunk_mode=current_chunk_mode,
                    max_chunk_chars=args.max_chunk_chars,
                )
                
                # Save with timestamp
                import time
                output_path = _resolve_output_path(
                    f"output_{int(time.time())}.wav",
                    str(output_dir),
                    iter_suffix,
                )
                tts.save_audio(audio, sr, str(output_path))
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    elif args.input_file:
        # Batch mode
        print(f"\nBatch processing from {args.input_file}")
        
        with open(args.input_file, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]
        
        for i, text in enumerate(tqdm(lines, desc="Synthesizing")):
            try:
                # Check for speaker specification: "speaker|text"
                if '|' in text:
                    speaker, text = text.split('|', 1)
                    speaker = speaker.strip()
                else:
                    speaker = default_speaker
                
                audio, sr = tts.synthesize(
                    text, speaker,
                    sdp_ratio=args.sdp_ratio,
                    noise_scale=args.noise_scale,
                    noise_scale_w=args.noise_scale_w,
                    length_scale=args.length_scale,
                    chunk_mode=args.chunk_mode,
                    max_chunk_chars=args.max_chunk_chars,
                )
                
                output_path = _resolve_output_path(
                    f"{i:04d}.wav",
                    str(output_dir),
                    iter_suffix,
                )
                tts.save_audio(audio, sr, str(output_path))
                
            except Exception as e:
                print(f"Error processing line {i}: {e}")
        
        print(f"\nBatch processing complete. Outputs saved to {output_dir}")
    
    elif args.text:
        # Single text mode
        print(f"\nSynthesizing: {args.text}")
        print(f"Speaker: {default_speaker}")
        
        audio, sr = tts.synthesize(
            args.text, default_speaker,
            sdp_ratio=args.sdp_ratio,
            noise_scale=args.noise_scale,
            noise_scale_w=args.noise_scale_w,
            length_scale=args.length_scale,
            chunk_mode=args.chunk_mode,
            max_chunk_chars=args.max_chunk_chars,
        )
        
        output_path = _resolve_output_path(args.output, str(output_dir), iter_suffix)
        tts.save_audio(audio, sr, str(output_path))
    
    else:
        print("Please provide --text, --input_file, or --interactive")
        print("Example: python infer.py --checkpoint G_10000.pth --config config.json --text 'Xin chào'")
        print("         python infer.py --text 'Very long text...' --chunk_mode auto --max_chunk_chars 150")


if __name__ == "__main__":
    main()
