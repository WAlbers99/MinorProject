# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 21:21:43 2022

@author: Wesse
"""

import selfies as sf
import numpy as np
import tensorflow as tf

smiles_dataset = ["CCCCCCC", "CCCCC(C)C", "CCCC(C)CC", "CCCC(C)(C)C","CCC(C)C(C)C", "CC(C)CC(C)C","CCC(C)(C)CC","CC(C)C(C)(C)C"]
selfies_dataset = list(map(sf.encoder, smiles_dataset))


max_len = max(sf.len_selfies(s) for s in selfies_dataset)

alphabet = sf.get_alphabet_from_selfies(selfies_dataset)
alphabet.add("[nop]")

vocab_stoi = {symbol: idx for idx, symbol in enumerate(alphabet)}
vocab_itos = {idx: symbol for symbol, idx in vocab_stoi.items()}

for first in selfies_dataset:
    one_hot = np.array(sf.selfies_to_encoding(first, vocab_stoi, pad_to_len =max_len)[1])
    #print(label)
    one_hot = one_hot.reshape(one_hot.shape[1]*one_hot.shape[0])
    print(one_hot.shape)
    print()
    
