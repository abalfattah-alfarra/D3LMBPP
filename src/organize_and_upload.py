import pandas as pd
from pyarabic.araby import strip_tashkeel, normalize_ligature
from pyarabic.stem import ArabicLightStemmer
from farasa.pos import FarasaPOSTagger

# Initialize Arabic light stemmer
light_stemmer = ArabicLightStemmer()

# Initialize Farasa POS Tagger (you need to install Farasa libraries)
# pip install farasa
farasa_tagger = FarasaPOSTagger(interactive=True)

def process_arabic_word(word):
    # Normalize and remove diacritics
    normalized_word = normalize_ligature(strip_tashkeel(word))

    # Get the stem using light stemming
    stem = light_stemmer.light_stem(normalized_word)

    # Extract suffixes (if any) using light stemming results
    suffixes = light_stemmer.get_suffix()

    # Get Part-of-Speech (POS) tagging using Farasa
    pos_tags = farasa_tagger.tag(word)
    pos = pos_tags[0][1] if pos_tags else "Unknown"

    return {
        "Stem": stem,
        "Suffixes": suffixes if suffixes else "None",
        "POS": pos
    }

# Load the Excel file from D:\arabic
file_path = r"D:\\arabic\\ab1.xlsx"
data = pd.read_excel(file_path)

# Process each word in the DataFrame and update the columns
data[['Stem', 'Suffixes', 'POS']] = data['Word'].apply(lambda word: pd.Series(process_arabic_word(word)))

# Save the updated DataFrame back to Excel
output_path = r"D:\\arabic\\arabic_word_analysis_updated.xlsx"
data.to_excel(output_path, index=False)
print(f"Updated Excel file saved to: {output_path}")
