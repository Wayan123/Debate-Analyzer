from collections import Counter

# Read the emotion data from the file
# Specify the path to your emotion data file
file_path = 'emotion_data_mahfud.txt'
with open(file_path, 'r') as f:
    emotion_data = f.read().strip().split('\n')

# Count the occurrences of each emotion using Counter
emotion_count = Counter(emotion_data)

# Print the count of each emotion
for emotion, count in emotion_count.items():
    print(f"{emotion}: {count}")
