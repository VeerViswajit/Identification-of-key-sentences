import pandas as pd
import random

# Define topics and their keyphrases
topics = {
    "Nature": ["forest", "wildlife", "ecosystem", "biodiversity", "conservation"],
    "Travel": ["destination", "adventure", "culture", "itinerary", "landmarks"],
    "Science Fiction": ["futuristic", "aliens", "space travel", "technology", "parallel universe"],
    "Cooking": ["recipe", "ingredients", "cuisine", "flavor", "culinary"],
    "Music": ["melody", "harmony", "genre", "instrument", "composition"],
    "Fitness": ["exercise", "workout", "nutrition", "health", "well-being"],
    "History": ["ancient", "civilization", "war", "empire", "historical"],
    "Geography": ["continent", "river", "mountain", "climate", "desert"],
    "Literature": ["novel", "poetry", "drama", "author", "literary"],
    "Technology": ["innovation", "software", "hardware", "internet", "automation"]
}

# List of irrelevant phrases to be used
irrelevant_phrases = [
    "The sky is blue.", "Cats are great pets.", "Pizza is delicious.", "Mount Everest is the tallest mountain.",
    "The sun rises in the east.", "Shakespeare wrote many plays.", "Dolphins are intelligent creatures.",
    "Chocolate comes from cocoa.", "Bats are the only flying mammals.", "The Pacific Ocean is the largest ocean."
]

# Generate paragraphs with mixed keyphrases and irrelevant sentences
def generate_paragraphs(topic, keyphrases):
    paragraphs = []
    for i in range(10):  # Generate 10 paragraphs per topic
        if i < 2:  # 2 paragraphs with 5 keyphrases and 1 irrelevant
            kp_used = 5
            ir_used = 1
        elif i < 4:  # 2 paragraphs with 4 keyphrases and 2 irrelevant
            kp_used = 4
            ir_used = 2
        elif i < 6:  # 2 paragraphs with 3 keyphrases and 3 irrelevant
            kp_used = 3
            ir_used = 3
        elif i < 8:  # 2 paragraphs with 2 keyphrases and 4 irrelevant
            kp_used = 2
            ir_used = 4
        else:  # 2 paragraphs with 1 keyphrase and 5 irrelevant
            kp_used = 1
            ir_used = 5

        key_sentences = [f"{kp.capitalize()} is essential for {topic.lower()}." for kp in random.sample(keyphrases, kp_used)]
        ir_sentences = random.sample(irrelevant_phrases, ir_used)
        all_sentences = key_sentences + ir_sentences
        random.shuffle(all_sentences)
        paragraph = ' '.join(all_sentences)
        keyphrase_list = '. '.join([kp.split()[0].lower() for kp in key_sentences])  # Extracting keyphrases correctly
        paragraphs.append([paragraph, keyphrase_list])
    return paragraphs

# Generating data for all topics
all_paragraphs = []
for topic, keyphrases in topics.items():
    all_paragraphs.extend(generate_paragraphs(topic, keyphrases))

# Create DataFrame and save to Excel
df = pd.DataFrame(all_paragraphs, columns=['Paragraph', 'Keyphrases'])
output_path = 'Generated_Keyphrases_Dataset_v2.xlsx'
df.to_excel(output_path, index=False)

print("Excel file has been generated and saved as 'Generated_Keyphrases_Dataset_v2.xlsx'")
