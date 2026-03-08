"""
Task Definitions
Defines available vocal tasks and their requirements.
"""


_TASKS = [
    {
        "id": "sustained_vowel",
        "display_name": "Sustained Vowel",
        "instruction": "Take a deep breath and say 'ahh' at a comfortable pitch for as long as you can.",
        "min_duration_sec": 5,
        "silence_before_sec": 3,
        "conditions_screened": ["benign_lesion"],
        "quality_gates": ["silence_region", "snr", "clipping", "voiced_duration", "agc"]
    },
    {
        "id": "free_speech",
        "display_name": "Free Speech",
        "instruction": "Tell us about your day or describe a place you'd like to visit. Speak naturally for about 30 seconds.",
        "min_duration_sec": 20,
        "silence_before_sec": 3,
        "conditions_screened": ["benign_lesion"],
        "quality_gates": ["silence_region", "snr", "clipping", "voiced_duration", "agc"]
    },
    {
        "id": "reading_passage",
        "display_name": "Reading Passage",
        "instruction": "Read the following passage aloud at your normal speaking pace.",
        "min_duration_sec": 15,
        "silence_before_sec": 3,
        "conditions_screened": ["benign_lesion"],
        "quality_gates": ["silence_region", "snr", "clipping", "voiced_duration", "agc"]
    },
    {
        "id": "cough",
        "display_name": "Voluntary Cough",
        "instruction": "Cough naturally three times with a short pause between each.",
        "min_duration_sec": 5,
        "silence_before_sec": 3,
        "conditions_screened": ["benign_lesion"],
        "quality_gates": ["silence_region", "snr", "clipping", "voiced_duration", "agc"]
    }
]


def get_all_tasks() -> list:
    """Return all supported vocal tasks."""
    return _TASKS


def get_task_by_id(task_id: str) -> dict:
    """Get a specific task by its ID."""
    for task in _TASKS:
        if task["id"] == task_id:
            return task
    return None
