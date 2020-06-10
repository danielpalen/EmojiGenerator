
'''
Visit https://unicode.org/emoji/charts/full-emoji-list.html
- to get a full list of emojis
- to understand the grouping of emojis
'''


# The 13 "standard" friendly emojis
# All should be available in all 4 databases
FACE_SMILING_EMOJIS = [
    f'grinning face', f'grinning face with big eyes', 'grinning face with smiling eyes',
    f'beaming face with smiling eyes', f'grinning squinting face', f'grinning face with sweat',
    f'rolling on the floor laughing', f'face with tears of joy', f'slightly smiling face',
    f'upside-down face', f'winking face', f'smiling face with smiling eyes', f'smiling face with halo'
]

# 8 emojis
# on unicode.org are 9, but "⊛ smiling face with tear" is not present in our database
FACE_AFFECTION_EMOJIS = [
    f'smiling face with hearts', f'smiling face with heart-eyes', f'star-struck',
    f'face blowing a kiss', f'kissing face', f'smiling face',
    f'kissing face with closed eyes', f'kissing face with smiling eyes'
]