
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

FACE_EMOJIS_HIGH = [
    f'smiling face with smiling eyes', f'face savoring food', f'relieved face',
    f'smirking face', f'kissing face with closed eyes', f'face with tongue',
    f'sleepy face', f'tired face', f'grimacing face', f'loudly crying face', f'face with open mouth', f'zany face',
    f'shushing face', f'face with hand over mouth', f'face vomiting', f'exploding head', f'',
    f'grinning face with sweat', f'grinning squinting face', f'smiling face with halo', f'winking face', f'neutral face',
    f'expressionless face', f'confused face', f'kissing face', f'face blowing a kiss', f'kissing face with smiling eyes', f'crying face',
    f'face with steam from nose', f'weary face', f'astonished face', f'sleeping face', f'dizzy face', f'face without mouth',
    f'face with medical mask', f'slightly frowning face', f'slightly smiling face', f'upside-down face', f'zipper-mouth face',
    f'money-mouth face', f'face with thermometer',f'nerd face', f'thinking face', f'nauseated face', f'drooling face', f'sneezing face',
    f'face with raised eyebrow', f'star-struck', f'smiling face with hearts', f'yawning face',
    f'partying face', f'woozy face', f'hot face', f'cold face', f'smiling face with smiling eyes', f'frowning face',
    f'full moon face'





]

FACE_EMOJIS_LOW = [
    f'full moon face',f'face with monocle',f'angry face with horns',f'relieved face',f'smiling face with heart-eyes',f'smiling face with sunglasses',f'smirking face',f'face with tongue',f'winking face with tongue',f'squinting face with tongue',f'disappointed face',
    f'worried face',f'grinning cat',f'grinning cat with smiling eyes',f'smiling cat with heart-eyes',f'cat with wry smile',f'kissing cat',f'crying cat',f'pouting cat',f'face with symbols on mouth',f'exploding head',f'smiling face with horns',f'neutral face',f'',
    f'expressionless face',f'smirking face',f'downcast face with sweat',f'pensive face',f'confounded face',f'angry face',f'pouting face',f'crying face',f'persevering face',f'sad but relieved face',f'frowning face with open mouth',f'anguished face',f'fearful face',
    f'anxious face with sweat',f'face screaming in fear',f'flushed face',f'sleeping face',f'face without mouth',
    f'face with rolling eyes',f'face with head-bandage',f'hugging face',f'cowboy hat face',f'nauseated face',
    f'drooling face',f'lying face',f'face with raised eyebrow',f'smiling face'
]