import os
import random
import math
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

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
        energy = energy - math.log(Q.get(bigram, epsilon))
    
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

def metropolis_step(perm, scrambled_text, P, Q, beta=1, old_energy=math.inf, old_states=dict()):
    """
    Performs a single step of the Metropolis-Hastings algorithm.

    Parameters:
        perm (dict): The current permutation.
        scrambled_text (str): The scrambled text.
        P (dict): The unigram probabilities.
        Q (dict): The bigram probabilities.
        beta (float): The inverse temperature. Default is 1.
        old_energy (float): The energy of the current permutation. Default is infinity.
        old_states (dict): The previously visited states. Default is an empty dictionary.

    Returns:
        dict: The new permutation.
        float: The energy of the new permutation.
        dict: The updated dictionary of previously visited states.
    """
    new_perm = propose_permutation(perm)
    
    if str(new_perm) in old_states:
        new_energy = old_states[str(new_perm)]
    else:
        new_energy = energy_function(new_perm, scrambled_text, P, Q)
        old_states[str(new_perm)] = new_energy

    E_delta = new_energy - old_energy

    if E_delta < 0 or random.random() < math.exp(-beta * E_delta):
        return new_perm, new_energy, old_states
    
    return perm, old_energy, old_states

def metropolis_hastings(scrambled_text, P, Q, beta=1, n=10**5, perm=None, return_chain=False):
    """
    Runs the Metropolis-Hastings algorithm to decode a scrambled text.

    Parameters:
        scrambled_text (str): The scrambled text.
        P (dict): The unigram probabilities.
        Q (dict): The bigram probabilities.
        beta (float): The inverse temperature. Default is 1.
        n (int): The length of the chain to be returned. Default is 1e5.
        perm (dict): The initial permutation. Default is None (which becomes identity).
        return_chain (bool): Whether to return the chain. Default is False.
    
    Returns:
        list: The chain of permutations.
        list: The chain of energies.
        dict: The best permutation found.
        float: The energy of the best permutation.
    """
    if perm is None:
        alphabet = list("abcdefghijklmnopqrstuvwxyz ")
        perm = {k: k for k in alphabet}

    energy = math.inf
    old_states = dict()

    best_perm = perm
    best_energy = energy

    if return_chain:
        state_chain = []
        energy_chain = []

        for _ in tqdm(range(n+10**4)):
            perm, energy, old_states = metropolis_step(perm, scrambled_text, P, Q, beta, energy, old_states)
            state_chain.append(perm)
            energy_chain.append(energy)

            if energy < best_energy:
                best_perm = perm
                best_energy = energy
        
        return state_chain[10**4:], energy_chain[10**4:], best_perm, best_energy

    for _ in tqdm(range(n)):
        perm, energy, old_states = metropolis_step(perm, scrambled_text, P, Q, beta, energy, old_states)

        if energy < best_energy:
            best_perm = perm
            best_energy = energy
    
    return None, None, best_perm, best_energy

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

    state_chain, energy_chain, best_perm, best_energy = metropolis_hastings(scrambled_text, P, Q, beta=0.5, n=10**6, perm=None, return_chain=True)

    best_perm = inv_perm(best_perm)
    decoded_text = "".join([best_perm.get(symbol, symbol) for symbol in scrambled_text])

    print(f"Scrambled text: \n{scrambled_text}\n")
    print(f"Best energy: \n{best_energy}\n")
    print(f"Decoded text: \n{decoded_text}")

    count = Counter([tuple(perm.items()) for perm in state_chain])
    count = count.most_common(len(count))

    pmf = []
    cdf = []

    for _, cnt in count:
        pmf.append(cnt / len(state_chain))
        cdf.append(sum(pmf))

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(count)), pmf)
    plt.xlabel("Permutation")
    plt.ylabel("Probability")
    plt.title("Probability Mass Function")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(count)), cdf)
    plt.xlabel("Permutation")
    plt.ylabel("Cumulative Probability")
    plt.title("Cumulative Distribution Function")
    plt.show()

    energy_chain = np.array(energy_chain)
    energy_chain = np.log(energy_chain)

    plt.figure(figsize=(10, 6))
    plt.plot(energy_chain)
    plt.xlabel("Iteration")
    plt.ylabel("Log Energy")
    plt.title("Log Energy vs. Iteration")
    plt.show()

if __name__ == "__main__":
    main()