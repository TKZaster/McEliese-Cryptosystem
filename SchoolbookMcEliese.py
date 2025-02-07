import numpy as np

def G_to_H(G):  # Converts a given generator matrix into its parity-check counterpart
    Q = G[:, G.shape[0]:]  # The non-padded parity matrix
    I_nk = np.eye(Q.shape[1], dtype=int)  # Identity matrix of size n - k
    H = np.concatenate((Q.T, I_nk), axis=1)  # The parity-check matrix
    return H

def gf2_inverse(matrix):
    from sympy import Matrix
    mat = Matrix(matrix.tolist())
    return np.array(mat.inv_mod(2)).astype(int)

# Definitions, input known matrices here:
G = np.array([[1, 0, 0, 0, 1, 1, 1],
              [0, 1, 0, 0, 0, 1, 1],
              [0, 0, 1, 0, 1, 0, 1],
              [0, 0, 0, 1, 1, 1, 0]])  # Generator Matrix
assert G.shape[1] > G.shape[0], "G must be a k x n matrix where n > k."

H = G_to_H(G)  # Parity-Check Matrix

P = np.array([[0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1],
              [1, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 1, 0, 0]])  # Permutation Matrix
assert P.shape[0] == P.shape[1], "P must be a square permutation matrix."

S = np.array([[1, 1, 0, 1],
              [1, 0, 0, 1],
              [0, 1, 1, 1],
              [1, 1, 0, 0]])  # Scrambler Matrix
assert S.shape[0] == S.shape[1] and S.shape[0] == G.shape[0], "S must be a square matrix of size k."

G_ = np.dot(np.dot(S, G) % 2, P) % 2  # Half of the public key (G', t) with private key (S, G, P)
print("-" * 14, "New Matrices", "-" * 14)
print(f"\nG'=\n {G_}\n")

# Encryption:
print("-" * 15, "Encryption", "-" * 15)
m = [int(char) for char in input("\nMessage to encrypt: ")]
assert len(m) == G.shape[0], f"Message must have length {G.shape[0]}."

mG_ = np.dot(m, G_) % 2  # Encoded vector
e = [int(char) for char in input("Error term: ")]
assert len(e) == G_.shape[1], f"Error term must have length {G_.shape[1]}."

c = (mG_ + e) % 2  # Ciphertext
print(f"\nEncrypted ciphertext: {c}\n")

# Decryption:
print("-" * 15, "Decryption", "-" * 15)
P_1 = gf2_inverse(P)  # Inverse of permutation matrix
c_ = np.dot(c, P_1) % 2  # This is c'
syndrome = np.dot(H, c_.T) % 2
print(f"\nc' = {c_}")
print(f"\nSyndrome = {syndrome}")

if np.all(syndrome == 0):
    print("Zero-state syndrome: no bits to correct!")
else:
    print("\nComparing to matrix H...\n")
    for i in range(H.shape[1]):  # Compares syndrome to parity-check matrix
        column = H[:, i]
        print(f"{syndrome} = {column}?")
        if np.array_equal(column, syndrome):
            c_[i] = 0 if c_[i] == 1 else 1
            print("Match!")
            print(f"\nBit {i + 1} flipped: {c_}")
            break
    else:
        print("Unable to correct error: syndrome not found in parity-check matrix.")

print("\nTrimming off parity bits...")
c_Trim = c_[:(H.shape[1] - H.shape[0])]
print(f"\nDecoded message: {c_Trim}")

print("\nRecovering original message...")
S_1 = gf2_inverse(S)  # Inverse of scrambler matrix
m_recovered = np.dot(c_Trim, S_1) % 2
print(f"\nOriginal message: {m_recovered}\n")
