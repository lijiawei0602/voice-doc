from __future__ import annotations

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from uuid import uuid4

from app.core.config import get_settings
from app.core.exceptions import AppError, ERRORS, raise_app_error
from app.core.logging import get_logger

logger = get_logger(__name__)


def probe_duration_seconds(audio_path: Path) -> float:
    try:
        duration = librosa.get_duration(path=str(audio_path))
        return duration
    except Exception as exc:
        logger.error("获取音频时长失败: %s", exc)
        raise AppError(ERRORS["AUDIO_CORRUPTED"]) from exc


def butter_highpass(cutoff, sr, order=5):
    from scipy.signal import butter
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_lowpass(cutoff, sr, order=5):
    from scipy.signal import butter
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def apply_filter(data, b, a):
    from scipy.signal import filtfilt
    y = filtfilt(b, a, data)
    return y


def preprocess_audio(input_path: Path) -> tuple[Path, float]:
    settings = get_settings()
    duration = probe_duration_seconds(input_path)
    if duration > settings.max_audio_duration_seconds:
        raise_app_error("AUDIO_TOO_LONG")

    output_path = settings.temp_dir / f"preprocessed_{uuid4().hex}.wav"

    logger.info("开始音频预处理: %s", input_path)
    
    try:
        # Load audio
        y, sr = librosa.load(str(input_path), sr=settings.sample_rate, mono=True)
        
        # Apply filters
        if settings.denoise_enabled or True:  # Apply highpass and lowpass always
            # Highpass filter at 200 Hz
            b_hp, a_hp = butter_highpass(200, sr)
            y = apply_filter(y, b_hp, a_hp)
            
            # Lowpass filter at 3000 Hz
            b_lp, a_lp = butter_lowpass(3000, sr)
            y = apply_filter(y, b_lp, a_lp)
        
        # Save as wav
        sf.write(str(output_path), y, sr)
        
        return output_path, duration
        
    except Exception as exc:
        logger.error("音频预处理失败: %s", exc)
        raise AppError(ERRORS["AUDIO_CORRUPTED"]) from exc
