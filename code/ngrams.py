import os
from nltk.util import ngrams
from collections import Counter
import string
import pandas as pd
import random
from tqdm import tqdm
import re

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

def load_ngrams(file_path, ngram_type, n):
    """
    Load the n-grams from the file.

    Parameters:
        file_path (str): The path to the file containing the text.
        ngram_type (str): The type of n-grams to load. Must be 'char' or 'word'.
        n (int): The size of the n-grams.

    Returns:
        pd.DataFrame: A DataFrame containing the n-grams and their counts.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    text = remove_gutenberg_header_footer(text)
    counts = get_ngrams(text, ngram_type, n)

    df = pd.DataFrame(list(counts.items()), columns=[f'{n}-gram', 'count'])
    df['frequency'] = df['count'] / df['count'].sum()
    df = df.sort_values(by='count', ascending=False, ignore_index=True)

    return df

def remove_gutenberg_header_footer(text):
    """
    Remove the Project Gutenberg header and footer from the text.

    Parameters:
        text (str): The text to process.

    Returns:
        str: The text without the Project Gutenberg header and footer.
    """
    start_marker = r"\*\*\* START OF THE PROJECT GUTENBERG .* \*\*\*"
    end_marker = r"\*\*\* END OF THE PROJECT GUTENBERG .* \*\*\*"

    start = text.find(start_marker)
    end = text.find(end_marker)

    if start != -1 and end != -1:
        text = text[start + len(start_marker):end]

    return text.strip()

def get_ngrams(text, ngram_type, n):
    """
    Get the n-grams and counts from the text.

    Parameters:
        text (str): The text to process.
        ngram_type (str): The type of n-grams to get. Must be 'char' or 'word'.
        n (int): The size of the n-grams.
    
    Raises:
        ValueError: If ngram_type is not 'word' or 'char'.    
    
    Returns:
        Counter: A Counter object with the n-grams and their counts.
    """
    if ngram_type == 'word':
        alphabet = set(string.ascii_lowercase + " ")
        text = text.lower()
        text = "".join(a for a in text if a in alphabet)

        tokens = nltk.word_tokenize(text)
        ngrams_list = list(ngrams(tokens, n))
    elif ngram_type == 'char':
        alphabet = set(string.ascii_lowercase)
        text = text.lower()
        text = "".join(a for a in text if a in alphabet)
    
        ngrams_list = [text[i:i+n] for i in range(len(text) - n + 1)]
    else:
        raise ValueError("ngram_type must be 'word' or 'char'")
    
    return Counter(ngrams_list)

def generate_text(df, type, n, length):
    """
    Generate text using the n-grams from the DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the n-grams and their counts.
        type (str): The type of n-grams. Must be 'word' or 'char'.
        n (int): The size of the n-grams.
        length (int): The length of the text to generate.
    
    Raises:
        ValueError: If type is not 'word' or 'char'.    

    Returns:
        str: The generated text.
    """
    if type == 'word':
        ngrams_list = df[f'{n}-gram'].tolist()
        counts = df['count'].tolist()
    elif type == 'char':
        ngrams_list = df[f'{n}-gram'].tolist()
        counts = df['count'].tolist()
    else:
        raise ValueError("type must be 'word' or 'char'")

    if type == 'word':
        if n == 1:
            text_tokens = []
            for _ in tqdm(range(length)):
                next_ngram = random.choices(ngrams_list, weights=counts)[0]
                text_tokens.append(next_ngram[0])
            text = " ".join(text_tokens)
        else:
            text_tokens = list(random.choice(ngrams_list))
            for _ in tqdm(range(length - n)):
                prefix = tuple(text_tokens[-(n-1):])
                next_ngrams = [ngram for ngram in ngrams_list if ngram[:n-1] == prefix]
                if not next_ngrams:
                    break
                next_ngram = random.choices(next_ngrams, weights=[counts[ngrams_list.index(ngram)] for ngram in next_ngrams])[0]
                text_tokens.append(next_ngram[-1])
            text = " ".join(text_tokens)
    else:
        if n == 1:
            text = ""
            for _ in tqdm(range(length)):
                next_ngram = random.choices(ngrams_list, weights=counts)[0]
                text += next_ngram[0]
        else:
            text = random.choice(ngrams_list)
            for _ in tqdm(range(length - n)):
                next_ngrams = [ngram for ngram in ngrams_list if ngram.startswith(text[-(n-1):])]
                if not next_ngrams:
                    break
                next_ngram = random.choices(next_ngrams, weights=[counts[ngrams_list.index(ngram)] for ngram in next_ngrams])[0]
                text += next_ngram[-1]

    return text

def main():
    file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'pg28054.txt')

    char_onegram_counts = load_ngrams(file_path, 'char', 1)
    char_bigram_counts = load_ngrams(file_path, 'char', 2)

    print("Top 10 most common character 1-grams:")
    print(char_onegram_counts.head(10))
    print("\nTop 10 most common character 2-grams:")
    print(char_bigram_counts.head(10))

    word_onegram_counts = load_ngrams(file_path, 'word', 1)
    word_bigram_counts = load_ngrams(file_path, 'word', 2)

    print("\nTop 10 most common word 1-grams:")
    print(word_onegram_counts.head(10))
    print("\nTop 10 most common word 2-grams:")
    print(word_bigram_counts.head(10))

    generated_text = generate_text(char_bigram_counts, 'char', 2, 100)
    print("\nGenerated text:")
    print(generated_text)

    generated_text = generate_text(word_bigram_counts, 'word', 2, 10)
    print("\nGenerated text:")
    print(generated_text)

if __name__ == "__main__":
    main()
