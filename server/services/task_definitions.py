"""
Task Definitions
Defines available vocal tasks and their requirements.
"""


def get_all_tasks() -> list:
    """Return all supported vocal tasks."""
    return [
        {
            "id": "sustained_vowel",
            "display_name": "Sustained Vowel",
            "instruction": "Take a deep breath and say 'ahh' at a comfortable pitch for as long as you can.",
            "min_duration_sec": 5,
            "silence_before_sec": 3,
            "conditions_screened": ["vocal_fold_paralysis", "parkinsons"]
        },
        {
            "id": "free_speech",
            "display_name": "Free Speech",
            "instruction": "Tell us about your day or describe a place you'd like to visit. Speak naturally for about 30 seconds.",
            "min_duration_sec": 20,
            "silence_before_sec": 3,
            "conditions_screened": ["depression", "parkinsons"]
        },
        {
            "id": "reading_passage",
            "display_name": "Reading Passage",
            "instruction": "Read the following passage aloud at your normal speaking pace.",
            "min_duration_sec": 15,
            "silence_before_sec": 3,
            "conditions_screened": ["vocal_fold_paralysis", "parkinsons"]
        },
        {
            "id": "cough",
            "display_name": "Voluntary Cough",
            "instruction": "Cough naturally three times with a short pause between each.",
            "min_duration_sec": 5,
            "silence_before_sec": 3,
            "conditions_screened": ["copd"]
        },
        {
        "id": "fimo",
        "display_name": "Forced Inhale Mouth Open (FIMO)",
        "instruction": "Exhale completely, then inhale quickly through your mouth as if you are trying to catch your breath. Record three of these breaths.",
        "min_duration_sec": 5,
        "silence_before_sec": 3,
        "conditions_screened": ["airway_stenosis", "stridor"]
    },
    {
        "id": "deep_breath",
        "display_name": "Deep Breath",
        "instruction": "Take five big breaths in and out through your mouth.",
        "min_duration_sec": 15,
        "silence_before_sec": 3,
        "conditions_screened": ["airway_stenosis", "stridor", "respiratory_disorders"]
    },
    {
        "id": "cinderella_story",
        "display_name": "Cinderella Story Retelling",
        "instruction": "Please retell the story of Cinderella in your own words with as much detail as you can remember.",
        "min_duration_sec": 30,
        "silence_before_sec": 3,
        "conditions_screened": ["alzheimers", "parkinsons", "dementia"]
    },
    {
        "id": "picture_description",
        "display_name": "Picture Description",
        "instruction": "Look at the provided picture (such as the Cookie Theft) and describe everything you see happening in it.",
        "min_duration_sec": 20,
        "silence_before_sec": 3,
        "conditions_screened": ["alzheimers", "neurodegenerative_disorders"]
    },
    {
        "id": "stroop_test",
        "display_name": "Word-Color Stroop Test",
        "instruction": "Name the color of the ink the words are printed in, rather than reading the words themselves, as quickly as possible.",
        "min_duration_sec": 20,
        "silence_before_sec": 3,
        "conditions_screened": ["neurological_disorders", "depression"]
    },
    {
        "id": "diadochokinesis",
        "display_name": "Diadochokinetic Rate",
        "instruction": "Take a deep breath and repeat the sounds 'pa-ta-ka' as quickly and clearly as you can.",
        "min_duration_sec": 5,
        "silence_before_sec": 3,
        "conditions_screened": ["depression", "motor_speech_disorders"]
    }
]
