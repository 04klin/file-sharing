// Suppress wornings (especially for early development)
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use hex::encode;
use rand::SeedableRng;
use std::{
  collections::hash_map::DefaultHasher,
  hash::{Hash, Hasher},
  mem,
};
/*
We'll use Rust's built-in hashing which returns a u64 type.
This alias just helps us understand when we're treating the number as a hash
*/
pub type HashValue = u64;

/// Helper function that makes the hashing interface easier to understand.
pub fn hash<T: Hash>(t: &T) -> HashValue {
  let mut s = DefaultHasher::new();
  t.hash(&mut s);
  s.finish()
}

/*
Given a vector of data blocks this function adds padding blocks to the end
until the length is a power of two which is needed for Merkle trees.
The padding value should be the empty string "".
*/
pub fn pad_base_layer(blocks: &mut Vec<&str>) {
  let n = blocks.len();

  if n != 0 && (n & (n - 1)) == 0 {
    return
  }
  // Finds next power of two
  let padding_length = blocks.len().next_power_of_two();

  blocks.resize(padding_length, "");
}

/*
Helper function to combine two hashes and compute the hash of the combination.
This will be useful when building the intermediate nodes in the Merkle tree.

Our implementation will hex-encode the hashes (as little-endian uints) into strings, concatenate
the strings, and then hash that string.
*/
pub fn concatenate_hash_values(left: HashValue, right: HashValue) -> HashValue {
  let combined = format!("{}{}", encode(left.to_le_bytes()), encode(right.to_le_bytes()));
  hash(&combined)
}

// calculate merkle root helper function
fn calculate_merkle_root_helper(hashes: Vec<HashValue>) -> HashValue {
  match hashes.len() {
    0 => 0,
    1 => hashes[0],
    _ => {
      let mut parent_level_hashes = Vec::new();

      for pair in hashes.chunks(2) {
        match pair.len() {
          1 => parent_level_hashes.push(pair[0]),
          _ => parent_level_hashes.push(concatenate_hash_values(pair[0], pair[1])),
        }
      }

      // Recursing on the upper level
      calculate_merkle_root_helper(parent_level_hashes)
    }
  }
}

/*
Calculates the Merkle root of a sentence. We consider each word in the sentence to
be one block. Words are separated by one or more spaces.

Example:
Sentence: "You trust me, right?"
"You", "trust", "me," "right?"
Notice that the punctuation like the comma and exclamation point are included in the words
but the spaces are not.
*/
pub fn calculate_merkle_root(sentence: &str) -> HashValue {
  // Separate sentence by whitespace
  let mut words: Vec<&str> = sentence.split_whitespace().collect();

  // Number of hashes is a 2^k number
  pad_base_layer(&mut words);

  let hashes: Vec<HashValue> = words.iter().map(|word| hash(word)).collect();

  println!("Hashes len:: {:?}", hashes);

  calculate_merkle_root_helper(hashes)
}

/*
A representation of a sibling node along the Merkle path from the data
to the root. It is necessary to specify which side the sibling is on
so that the hash values can be combined in the same order.
*/
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum SiblingNode {
  Left(HashValue),
  Right(HashValue),
}

/// A proof is just an alias for a vec of sibling nodes.
pub type MerkleProof = Vec<SiblingNode>;

/*
Generates a Merkle proof that one particular word is contained
in the given sentence. You provide the sentence and the index of the word
which you want a proof.

This makes it so that we can make sure nodes aren't tampered with

Panics if the index is beyond the length of the sentence.

Example: I want to prove that the word "trust" is in the sentence "You trust me, right?"
So I call generate_proof("You trust me, right?", 1)
And I get back the merkle root and list of intermediate nodes from which the
root can be reconstructed.
*/
pub fn generate_proof(sentence: &str, index: usize) -> (HashValue, MerkleProof) {
  // Preparation of leaf nodes
  let mut words: Vec<&str> = sentence.split_whitespace().collect();
  pad_base_layer(&mut words);

  let mut hashes: Vec<HashValue> = words.iter().map(|word| hash(word)).collect();
  
  // Initialization for Proof Generation
  let mut proof = Vec::new();
  let mut idx = index;

  // Climb the merkle tree
  while hashes.len() > 1 {
    // Identify sibling and direction
    let is_right_sibling = idx % 2 == 0;
    let sibling_idx = if is_right_sibling { idx + 1 } else { idx - 1 };

    if is_right_sibling {
      proof.push(SiblingNode::Right(hashes[sibling_idx]));
    } else {
      proof.push(SiblingNode::Left(hashes[sibling_idx]));
    }

    idx /= 2;

    let mut next_level = Vec::new();
    for pair in hashes.chunks(2) {
      next_level.push(concatenate_hash_values(pair[0], pair[1]));
    }
    hashes = next_level
  }
  (hashes[0], proof)
}

/*
Checks whether the given word is contained in a sentence, without knowing the whole sentence.
Rather we only know the merkle root of the sentence and a proof.
*/
pub fn validate_proof(root: &HashValue, word: &str, proof: MerkleProof) -> bool {
  let mut hash = hash(&word);

  for node in proof {
    hash = match node {
      SiblingNode::Left(sibling_hash) => concatenate_hash_values(sibling_hash, hash),
      SiblingNode::Right(sibling_hash) => concatenate_hash_values(hash, sibling_hash),
    }
  }

  // Return if the hash matches the root
  hash == *root
}

/*
A compact Merkle multiproof is used to prove multiple entries in a Merkle tree in a highly
space-efficient manner.
*/
#[derive(Debug, PartialEq, Eq)]
pub struct CompactMerkleMultiProof {
  // The indices requested in the initial proof generation
  pub leaf_indices: Vec<usize>,
  // The additional hashes necessary for computing the proof, given in order from
  // lower to higher index, lower in the tree to higher in the tree.
  pub hashes: Vec<HashValue>,
}

/*
Generate a compact multiproof that some words are contained in the given sentence. Returns the
root of the merkle tree, and the compact multiproof. You provide the words at `indices` in the
same order as within `indices` to verify the proof. `indices` is not necessarily sorted.

Panics if any index is beyond the length of the sentence, or any index is duplicated.

## Explanation

To understand the compaction in a multiproof, see the following merkle tree. To verify a proof
for the X's, only the entries marked with H are necessary. The rest can be calculated. Then, the
hashes necessary are ordered based on the access order. The H's in the merkle tree are marked
with their index in the output compact proof.

```text
                                     O            
                                  /     \           
                               O           O     
                             /   \       /   \     
                            O    H_1   H_2    O  
                           / \   / \   / \   / \
                          X  X  O  O  O  O  X  H_0
```

The proof generation process would proceed similarly to a normal merkle proof generation, but we
need to keep track of which hashes are known to the verifier by a certain height, and which need
to be given to them.

In the leaf-node layer, the first pair of hashes are both
known, and so no extra data is needed to go up the tree.  In the next two pairs of hashes,
neither are known, and so the verifier does not need them. In the last set, the verifier only
knows the left hash, and so the right hash must be provided.

In the second layer, the first and fourth hashes are known. The first pair is missing the right
hash, which must be included in the proof. The second pair is missing the left hash, which also
must be included.

In the final layer before the root, both hashes are known to the verifier, and so no further
proof is needed.

The final proof for this example would be
```ignore
CompactMerkleMultiProof {
    leaf_indices: [0, 1, 6],
    hashes: [H_0, H_1, H_2]
}
```
*/
pub fn generate_compact_multiproof(
  sentence: &str,
  indices: Vec<usize>,
) -> (HashValue, CompactMerkleMultiProof) {

  /*
  For each of the indices, takes the index of its immediate neightbor,
  and stored the given element index and the neighboring index as a pair
  of indices

  Looks at the difference between pair indices and indices
  Appends the hash for given values to the multiproof
  */

  let words: Vec<&str> = sentence.split_whitespace().collect();

  // Panics if any index is beyond the length of the sentence, or any index is duped.
  for &index in &indices {
    if index >= words.len() {
      panic!("Index {} is out of bounds", index);
    }
  }

  // Hashes the words into leaf nodes
  let mut nodes: Vec<HashValue> = words.iter().map(|&word| hash(&word)).collect();
  let mut hashes = Vec::new();
  let mut leaf_indices = indices.clone();

  // Builds the tree, layer by layer, remembering which hashes are needed for the proof
  while nodes.len() > 1 {
    let mut next_level_nodes = Vec::new();
    let mut next_level_indices = Vec::new();

    for i in 0..nodes.len() / 2 {
      let left_child = i * 2;
      let right_child = left_child + 1;

      // Compute the new hash for this node
      let new_hash = concatenate_hash_values(nodes[left_child], nodes[right_child]);
      next_level_nodes.push(new_hash);

      // If either child is in the index set, this node's index needs to be in the next level's index set
      if leaf_indices.contains(&left_child) || leaf_indices.contains(&right_child) {
        next_level_indices.push(i);
      }

      // If the right child's hash is known but the left child's is not, include the right child's hash in the proof
      if leaf_indices.contains(&right_child) && !leaf_indices.contains(&left_child) {
        hashes.push(nodes[left_child]);
      }
      // Vice versa
      if leaf_indices.contains(&left_child) && !leaf_indices.contains(&right_child) {
        hashes.push(nodes[right_child]);
      }
    }

    nodes = next_level_nodes;
    leaf_indices = next_level_indices;
  }

  // The root of the tree is the remaining node
  let root = nodes[0];

  // The proof consists of the original indices and the hashes we collected
  let proof = CompactMerkleMultiProof {
    leaf_indices: indices,
    hashes,
  };

  (root, proof)
}

/*
Validate a compact merkle multiproof to check whether a list of words is contained in a sentence, based on the merkle root of the sentence.
The words must be in the same order as the indices passed in to generate the multiproof.
Duplicate indices in the proof are rejected by returning false.
*/
pub fn validate_compact_multiproof(
  root: &HashValue,
  words: Vec<&str>,
  proof: CompactMerkleMultiProof,
) -> bool {
  /*
  Step 1. reconstruct the merkle tree from the given words and proof:
   - for each indices take the index of its immediate neighbor
   and store the given element index and the neighboring index as a pair of indices
   - check for duplicate pairs
   - if there are no leaf_indices
   - hash the corresponding value
   - we take the even numbers of from the pairs an divide them by two
   - repeat

  Step 2. compare the given root with the root of the reconstructed tree
  */

  // Step 1: Reconstruct the Merkle Tree from the given words and proof
  let mut nodes: Vec<HashValue> = words.iter().map(|&word| hash(&word)).collect();
  let mut leaf_indices = proof.leaf_indices;
  let mut proof_hashes = proof.hashes;

  // Check for duplicate indices
  let mut seen = std::collections::HashSet::new();
  for &index in &leaf_indices {
    if !seen.insert(index) {
      return false;
    }
  }

  // Check if the lengths of leaf_indices and nodes don't match
  if leaf_indices.len() != nodes.len() {
    return false;
  }

  // Max leaf index to control loop
  let mut max_leaf_index: usize = *leaf_indices.iter().max().unwrap();

  // Process the levels of the Merkle Tree
  while !proof_hashes.is_empty() || max_leaf_index > 1 {
    let mut next_level_nodes = Vec::new();
    let mut next_level_indices = Vec::new();

    max_leaf_index = *leaf_indices.iter().max().unwrap_or(&0);

    for i in 0..=max_leaf_index {
      let left_child = i * 2;
      let right_child = left_child + 1;

      match (leaf_indices.contains(&left_child), leaf_indices.contains(&right_child)) {
        (false, false) => continue,
        (true, false) => {
          // Ensure the left child index is valid
          if let Some(left_child_index) = leaf_indices.iter().position(|&x| x == left_child) {
            let left_hash = nodes[left_child_index];
            if !proof_hashes.is_empty() {
              let right_hash = proof_hashes.remove(0);
              next_level_nodes.push(concatenate_hash_values(left_hash, right_hash));
            } else {
              next_level_indices.push(i);
            }
          } else {
            // Invalid left child index
            return false;
          }
        },
        (false, true) => {
          if let Some(right_child_index) = leaf_indices.iter().position(|&x| x == right_child) {
            if !proof_hashes.is_empty() {
              let left_hash = proof_hashes.remove(0);
              let right_hash = nodes[right_child_index];
              next_level_nodes.push(concatenate_hash_values(left_hash, right_hash));
            } else {
              return false;
            }
          } else {
            return false;
          }
        },
        (true, true) => {
          if let (Some(left_child_index), Some(right_child_index)) = (
            leaf_indices.iter().position(|&x| x == left_child),
            leaf_indices.iter().position(|&x| x == right_child),
          ) {
            if left_child_index < nodes.len() && right_child_index < nodes.len() {
              let left_hash = nodes[left_child_index];
              let right_hash = nodes[right_child_index];
              next_level_nodes.push(concatenate_hash_values(left_hash, right_hash));
            } else {
              return false;
            }
          } else {
            return false;
          }
        },
      };
      next_level_indices.push(i);
    }
    nodes = next_level_nodes;
    leaf_indices = next_level_indices;
  }
  nodes[0] == *root
}

/*
Now that we have a normal and compact method to generate proofs, let's compare how
space-efficient the two are. The two functions below will be helpful for answering the questions
in the readme.

Generate a space-separated string of `n` random 4-letter words. Use of this function is not
mandatory.
*/
pub fn string_of_random_words(n: usize) -> String {
  let mut ret = String::new();
  for i in 0..n {
    ret.push_str(random_word::gen_len(4).unwrap());
    if i != n - 1 {
      ret.push(' ');
    }
  }
  ret
}

/*
Given a string of words, and the length of the words from which to generate proofs, generate
proofs for `num_proofs` random indices in `[0, length)`.  Uses `rng_seed` as the rng seed, if
replicability is desired.

Return the size of the compact multiproof, and then the combined size of the standard merkle proofs.

This function assumes the proof generation is correct, and does not validate them.
*/
pub fn compare_proof_sizes(
  words: &str,
  length: usize,
  num_proofs: usize,
  rng_seed: u64,
) -> (usize, usize) {
  assert!(
    num_proofs <= length,
    "Cannot make more proofs than available indices!"
  );

  let mut rng = rand::rngs::SmallRng::seed_from_u64(rng_seed);
  let indices = rand::seq::index::sample(&mut rng, length, num_proofs).into_vec();
  let (_, compact_proof) = generate_compact_multiproof(words, indices.clone());
  // Manually calculate memory sizes
  let compact_size = mem::size_of::<usize>() * compact_proof.leaf_indices.len()
    + mem::size_of::<HashValue>() * compact_proof.hashes.len()
    + mem::size_of::<Vec<usize>>() * 2;

  let mut individual_size = 0;
  for i in indices {
    let (_, proof) = generate_proof(words, i);
    individual_size +=
      mem::size_of::<Vec<usize>>() + mem::size_of::<SiblingNode>() * proof.len();
  }

  (compact_size, individual_size)
}

#[test]
#[ignore]
fn student_test_to_compare_sizes() {
  // Maybe write a test here to compare proof sizes in order to get answers to the following
  // questions.

let sentence = "Here is a sentence with some words I want to test.";

let words: Vec<&str> = sentence.split_whitespace().collect();
println!("Words len: {:?}", words.len());

let length = 11;
let num_proofs = 3;
let rng_seed = 12345678;

let (compact_size, individual_size) = compare_proof_sizes(&sentence, length, num_proofs, rng_seed);
println!("Compact size: {}", compact_size);
println!("Individual size: {}", individual_size);

assert!(compact_size <= individual_size);
}

/// An answer to the below short answer problems
#[derive(PartialEq, Debug)]
pub struct ShortAnswer {
  /// The answer to the problem
  pub answer: usize,
  /// The explanation associated with an answer. This should be 1-3 sentences. No need to make it
  /// too long!
  pub explanation: String,
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn pad_base_layer_sanity_check() {
    let mut data = vec!["a", "b", "c"];
    let expected = vec!["a", "b", "c", ""];
    pad_base_layer(&mut data);
    assert_eq!(expected, data);
  }

  #[test]
  fn concatenate_hash_values_sanity_check() {
    let left = hash(&"a");
    let right = hash(&"b");
    assert_eq!(13491948173500414413, concatenate_hash_values(left, right));
  }

  #[test]
  fn calculate_merkle_root_sanity_check() {
    let sentence = "You trust me, right?";
    assert_eq!(4373588283528574023, calculate_merkle_root(sentence));
  }

  #[test]
  fn proof_generation_sanity_check_2() {
    let sentence = "apex rite gite mite gleg meno merl nard bile ills hili";
    generate_proof(sentence, 1);
  }

  #[test]
  fn proof_generation_sanity_check() {
    let sentence = "You trust me, right?";
    let expected = (
      4373588283528574023,
      vec![
        SiblingNode::Left(4099928055547683737),
        SiblingNode::Right(2769272874327709143),
      ],
    );
    assert_eq!(expected, generate_proof(sentence, 1));
  }

  #[test]
  fn validate_proof_sanity_check() {
    let word = "trust";
    let root = 4373588283528574023;
    let proof = vec![
      SiblingNode::Left(4099928055547683737),
      SiblingNode::Right(2769272874327709143),
    ];
    assert!(validate_proof(&root, word, proof));
  }

  #[test]
  fn calculate_merkle_root_sanity_check_2() {
    let sentence = "You trust me?";
    assert_eq!(8656240816105094750, calculate_merkle_root(sentence));
  }

  #[test]
  fn generate_compact_multiproof_sanity_check() {
    let sentence = "Here's an eight word sentence, special for you.";
    let indices = vec![0, 1, 6];
    let expected = (
      14965309246218747603,
      CompactMerkleMultiProof {
        leaf_indices: vec![0, 1, 6],
        hashes: vec![
          1513025021886310739,
          7640678380001893133,
          5879108026335697459,
        ],
      },
    );
    assert_eq!(expected, generate_compact_multiproof(sentence, indices));
  }

  #[test]
  fn validate_compact_multiproof_sanity_check() {
    let proof = (
      14965309246218747603u64,
      CompactMerkleMultiProof {
        leaf_indices: vec![0, 1, 6],
        hashes: vec![
          1513025021886310739,
          7640678380001893133,
          5879108026335697459,
        ],
      },
    );
    let words = vec!["Here's", "an", "for"];
    assert_eq!(true, validate_compact_multiproof(&proof.0, words, proof.1));
  }

}


#[cfg(test)]
mod additional_tests {
  use super::*;

  #[test]
  fn test_single_word_merkle_root() {
    let sentence = "hello";
    let root = calculate_merkle_root(sentence);
    let expected_root = hash(&"hello");
    assert_eq!(root, expected_root);
  }

  #[test]
  fn test_empty_string_merkle_root() {
    let sentence = "";
    let root = calculate_merkle_root(sentence);
    let expected_root = hash(&"");
    assert_eq!(root, expected_root);
  }

  #[test]
  fn test_large_input_merkle_root() {
    let sentence = string_of_random_words(1024);
    let root = calculate_merkle_root(&sentence);
    assert_ne!(root, 0);
  }

  #[test]
  fn test_proof_for_last_word() {
    let sentence = "this is a test sentence with multiple words for merkle tree validation";
    let index = 11;
    let (root, proof) = generate_proof(sentence, index);
    let word = "validation";
    assert!(validate_proof(&root, word, proof));
  }

  #[test]
  fn test_generate_and_validate_proof() {
    let sentence = "the quick brown fox jumps over the lazy dog";
    for i in 0..9 {
      let (root, proof) = generate_proof(sentence, i);
      let word = sentence.split_whitespace().nth(i).unwrap();
      assert!(validate_proof(&root, word, proof));
    }
  }

  #[test]
  fn test_invalid_proof() {
    let sentence = "the quick brown fox jumps over the lazy dog";
    let (root, proof) = generate_proof(sentence, 0);
    let invalid_word = "invalid";
    assert!(!validate_proof(&root, invalid_word, proof));
  }

  #[test]
  fn test_multiproof_generation_and_validation() {
    let sentence = "this is another test sentence for multiproof validation";
    let indices = vec![1, 3, 6];
    let words = vec!["is", "test", "multiproof"];
    let (root, multiproof) = generate_compact_multiproof(sentence, indices.clone());
    assert!(validate_compact_multiproof(&root, words, multiproof));
  }

  #[test]
  fn test_invalid_multiproof() {
    let sentence = "this is another test sentence for multiproof validation";
    let indices = vec![1, 3, 6];
    let words = vec!["is", "test", "multiproof"];
    let (root, multiproof) = generate_compact_multiproof(sentence, indices.clone());
    let invalid_words = vec!["invalid", "multiproof", "random"];
    assert!(!validate_compact_multiproof(&root, invalid_words, multiproof));
  }

  #[test]
  fn test_multiproof_edge_case() {
    let sentence = "edge case with only one word";
    let indices = vec![0];
    let words = vec!["edge"];
    let (root, multiproof) = generate_compact_multiproof(sentence, indices.clone());
    assert!(validate_compact_multiproof(&root, words, multiproof));
  }

  #[test]
  fn test_multiproof_with_duplicates() {
    let sentence = "this sentence has duplicate words this sentence";
    let indices = vec![0, 4, 5, 6];
    let words = vec!["this", "words", "this", "sentence"];
    let (root, multiproof) = generate_compact_multiproof(sentence, indices.clone());
    assert!(!validate_compact_multiproof(&root, words, multiproof));
  }

  #[test]
  fn test_compare_proof_sizes() {
    let sentence = string_of_random_words(1024);
    let length = 1024;
    let num_proofs = 10;
    let rng_seed = 12345678;
    let (compact_size, individual_size) = compare_proof_sizes(&sentence, length, num_proofs, rng_seed);
    assert!(compact_size < individual_size);
  }

  #[test]
  fn test_calculate_generate_and_validate_proof() {
    // Step 1: Calculate Merkle root
    let sentence = "the quick brown fox jumps over the lazy dog";
    let root = calculate_merkle_root(sentence);
    assert_ne!(root, 0, "Merkle root should not be zero");

    // Step 2: Generate proof for a specific word
    let index = 3; // Let's choose the word "fox"
    let (generated_root, proof) = generate_proof(sentence, index);
    let word = "fox";
    
    // Ensure the generated root matches the calculated root
    assert_eq!(root, generated_root, "Generated root should match the calculated root");

    // Step 3: Validate the proof
    let is_valid = validate_proof(&root, word, proof);
    assert!(is_valid, "The proof should be valid for the word 'fox'");
  }
}