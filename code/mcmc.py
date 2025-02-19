import os
import random
import math
from tqdm import tqdm
from collections import Counter

import ngrams

def inv_perm(perm):
    """
    Inverts a permutation.

    Parameters:
        perm (dict): The permutation to invert.

    Returns:
        dict: The inverted permutation.
    """
    return {v: k for k, v in perm.items()}

def energy_function(perm, scrambled_text, P, Q):
    """
    Computes the energy of a permutation given a scrambled text and n-gram probabilities.

    Parameters:
        perm (dict): The permutation to evaluate.
        scrambled_text (str): The scrambled text.
        P (dict): The unigram probabilities.
        Q (dict): The bigram probabilities.

    Returns:
        float: The energy of the permutation.
    """
    sigma_inv = inv_perm(perm)
    
    decoded_text = [sigma_inv.get(symbol, symbol) for symbol in scrambled_text]

    epsilon = 1e-12
    energy = -math.log(P.get(decoded_text[0], epsilon))

    for j in range(len(decoded_text) - 1):
        bigram = (decoded_text[j], decoded_text[j+1])
        energy -= math.log(Q.get(bigram, epsilon))
    
    return energy

def propose_permutation(perm):
    """
    Proposes a new permutation by swapping two random elements.

    Parameters:
        perm (dict): The current permutation.

    Returns:
        dict: The proposed permutation.
    """
    new_perm = perm.copy()
    keys = list(new_perm.keys())
    i, j = random.sample(keys, 2)
    new_perm[i], new_perm[j] = new_perm[j], new_perm[i]

    return new_perm

def metropolis_step(perm, scrambled_text, P, Q, beta=1, old_energy=math.inf):
    """
    Performs a single step of the Metropolis-Hastings algorithm.

    Parameters:
        perm (dict): The current permutation.
        scrambled_text (str): The scrambled text.
        P (dict): The unigram probabilities.
        Q (dict): The bigram probabilities.
        beta (float): The inverse temperature. Default is 1.

    Returns:
        dict: The new permutation.
    """
    new_perm = propose_permutation(perm)
    
    new_energy = energy_function(new_perm, scrambled_text, P, Q)

    E_delta = new_energy - old_energy

    if E_delta < 0 or random.random() < math.exp(-beta * E_delta):
        return new_perm, new_energy
    
    return perm, old_energy

def metropolis_hastings(scrambled_text, P, Q, beta=1, n=10**5, perm=None):
    """
    Runs the Metropolis-Hastings algorithm to decode a scrambled text.

    Parameters:
        scrambled_text (str): The scrambled text.
        P (dict): The unigram probabilities.
        Q (dict): The bigram probabilities.
        beta (float): The inverse temperature. Default is 1.
        n (int): The length of the chain to be returned. Default is 1e5.
    
    Returns:
        list: The chain of permutations.
    """
    if perm is None:
        alphabet = list("abcdefghijklmnopqrstuvwxyz ")
        perm = {k: k for k in alphabet}

    energy = math.inf
    chain = []

    for _ in tqdm(range(n+10**4)):
        perm, energy = metropolis_step(perm, scrambled_text, P, Q, beta, energy)
        chain.append(perm)

    return chain[10**4:]

def main():
    file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'pg2554.txt')

    df1 = ngrams.load_ngrams(file_path, 'char', 1)
    df2 = ngrams.load_ngrams(file_path, 'char', 2)

    P = {row['1-gram']: row['frequency'] for _, row in df1.iterrows()}
    Q = {tuple(row['2-gram']): row['frequency'] for _, row in df2.iterrows()}

    alphabet = list("abcdefghijklmnopqrstuvwxyz ")
    permutation = alphabet.copy()
    random.shuffle(permutation)

    sigma = {k: v for k, v in zip(alphabet, permutation)}

    text = "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair, we had everything before us, we had nothing before us, we were all going direct to Heaven, we were all going direct the other way—in short, the period was so far like the present period, that some of its noisiest authorities insisted on its being received, for good or for evil, in the superlative degree of comparison only."
    
    alphabet = set(alphabet)
    text = text.lower()
    text = "".join(a for a in text if a in alphabet)

    scrambled_text = "".join([sigma.get(symbol, symbol) for symbol in text])

    chain = metropolis_hastings(scrambled_text, P, Q, beta=1, n=10**5)

    count = Counter([tuple(perm.items()) for perm in chain])
    perms = count.most_common(1)
    
    print(f"Scrambled text: \n{scrambled_text}\n")

    for perm, freq in perms:
        perm_inv = inv_perm(dict(perm))
        decoded_text = "".join([perm_inv.get(symbol, symbol) for symbol in scrambled_text])
        print(f"Frequency: \n{freq/len(chain)}\n")
        print(f"Decoded text: \n{decoded_text}")

if __name__ == "__main__":
    main()