// Suppress warnings (especially for early development)
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use hex::encode;
use rand::SeedableRng;
use std::{
  collections::{
    hash_map::DefaultHasher,
    HashSet,
    HashMap
  },
  hash::{Hash, Hasher},
  mem,
};
use serde::{Serialize, Deserialize};
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
    0 => hash(&""),
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
#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize)]
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

  pub leaf_count: usize,
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

  let mut words: Vec<&str> = sentence.split_whitespace().collect();

  let mut seen = HashSet::new();
  // Panics if any index is beyond the length of the sentence, or any index is a duplicate.
  for &index in &indices {
    if index >= words.len() {
      panic!("Index {} is out of bounds", index);
    }
    if !seen.insert(index) {
      panic!("Duplicate index {} found when generating compact multiproof", index);
    }
  }

  pad_base_layer(&mut words);

  let mut current_level_hashes: Vec<HashValue> = words.iter().map(|&word| hash(&word)).collect();

  let leaf_count = current_level_hashes.len();

  let mut proof_hashes = Vec::new();
  // Use a HashSet for efficient O(1) lookups of which indices are "known".
  let mut known_indices: HashSet<usize> = indices.iter().cloned().collect();

  // Build the tree from the bottom up.
  while current_level_hashes.len() > 1 {
    let mut next_level_hashes = Vec::new();
    let mut next_level_known_indices = HashSet::new();

    for i in 0..(current_level_hashes.len() / 2) {
      let left_child_idx = i * 2;
      let right_child_idx = left_child_idx + 1;

      let left_known = known_indices.contains(&left_child_idx);
      let right_known = known_indices.contains(&right_child_idx);

      // If we can compute a parent node because we know at least one child...
      if left_known || right_known {
        // ...then we mark the parent node as computable in the next level.
        next_level_known_indices.insert(i);

        // If we only know one child, the other one must be added to the proof.
        if left_known && !right_known {
          proof_hashes.push(current_level_hashes[right_child_idx]);
        } else if !left_known && right_known {
          proof_hashes.push(current_level_hashes[left_child_idx]);
        }
      }
      
      // We must compute all parent hashes to build the next level of the tree.
      let parent_hash = concatenate_hash_values(
        current_level_hashes[left_child_idx],
        current_level_hashes[right_child_idx],
      );
      next_level_hashes.push(parent_hash);
    }

    current_level_hashes = next_level_hashes;
    known_indices = next_level_known_indices;
  }

  let root = current_level_hashes[0];
  let proof = CompactMerkleMultiProof {
    leaf_indices: indices, // The original, unsorted indices.
    hashes: proof_hashes,
    leaf_count,
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
  // Basic sanity checks.
  if proof.leaf_indices.len() != words.len() {
      return false;
  }
  let mut seen_indices = HashSet::new();
  if !proof.leaf_indices.iter().all(|&i| seen_indices.insert(i)) {
      return false; // Reject proofs with duplicate indices.
  }

  let mut proof_hashes_iter = proof.hashes.iter();
  
  // Use a map to store known nodes: `index -> hash`.
  // Start by populating it with the leaf nodes from the words provided.
  let mut known_nodes: HashMap<usize, HashValue> = proof
      .leaf_indices
      .iter()
      .zip(words.iter())
      .map(|(&index, &word)| (index, hash(&word)))
      .collect();
  
  // Determine the original tree's leaf count
  let mut num_nodes_at_level = proof.leaf_count;

  // Climb the tree level by level until we reach the root.
  while num_nodes_at_level > 1 {
    let mut next_level_nodes = HashMap::new();
    for i in 0..(num_nodes_at_level / 2) {
      let left_idx = i * 2;
      let right_idx = left_idx + 1;

      let left_hash_opt = known_nodes.get(&left_idx);
      let right_hash_opt = known_nodes.get(&right_idx);

      let parent_hash = match (left_hash_opt, right_hash_opt) {
        // Case 1: We know both children. Combine them.
        (Some(l), Some(r)) => concatenate_hash_values(*l, *r),
        // Case 2: We know left, need right from proof.
        (Some(l), None) => {
          match proof_hashes_iter.next() {
            Some(r_proof) => concatenate_hash_values(*l, *r_proof),
            None => return false, // Proof is missing a hash.
          }
        }
        // Case 3: We know right, need left from proof.
        (None, Some(r)) => {
          match proof_hashes_iter.next() {
            Some(l_proof) => concatenate_hash_values(*l_proof, *r),
            None => return false, // Proof is missing a hash.
          }
        }
        // Case 4: We don't know either child, so this branch isn't needed for the proof.
        (None, None) => continue,
      };
      next_level_nodes.insert(i, parent_hash);
    }
    known_nodes = next_level_nodes;
    num_nodes_at_level /= 2;
  }
  
  // All proof hashes must have been consumed. If not, the proof is invalid.
  if proof_hashes_iter.next().is_some() {
    return false;
  }

  // Finally, the calculated root (at index 0) must match the provided root.
  match known_nodes.get(&0) {
    Some(calculated_root) => calculated_root == root,
    // This handles an edge case of an empty proof for an empty tree.
    None => proof.leaf_count <= 1 && hash(&"") == *root,
  }
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
  
  // More accurate size: sum of the size of elements in each vector.
  let compact_size = compact_proof.leaf_indices.len() * mem::size_of::<usize>()
    + compact_proof.hashes.len() * mem::size_of::<HashValue>();

  let mut individual_size = 0;
  for i in indices {
    let (_, proof) = generate_proof(words, i);
    // The size of a single proof is the size of its SiblingNode elements.
    individual_size += proof.len() * mem::size_of::<SiblingNode>();
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
        leaf_count: 8,
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
        leaf_count: 8,
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
    assert!(validate_compact_multiproof(&root, words, multiproof));
  }

  #[test]
  #[should_panic]
  fn test_multiproof_with_duplicates_panics() {
    let sentence = "this sentence has duplicate words this sentence";
    let indices_with_duplicates = vec![0, 4, 5, 0];
    generate_compact_multiproof(sentence, indices_with_duplicates);
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