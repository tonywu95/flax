# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Default Hyperparameter configuration."""

import ml_collections


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  # Path to load or store sentencepiece vocab file.
  config.vocab_path = "/checkpoint/ywu/data/preprocess_scripts/lean_gptf_sp_models/model_4000_bpe.model" 

  config.data_dir = "/checkpoint/ywu/data/ITP_DATA/lean_novel_lemma"

  # Vocabulary size if `vocab_path` is not given.
  config.vocab_size = 4000 

  config.max_corpus_chars = 10**7

  # Reverse the direction of translation.
  config.reverse_translation = False

  # Per host batch size for training.
  config.batch_size = 128 

  # Per host batch size for training.
  config.bucket_length = 256 

  # Beam size for inference.
  config.beam_size = 5 

  # Frequency of eval during training, e.g. every 1000 steps.
  config.eval_frequency = 1000

  # Number of train steps.
  config.num_train_steps = 200_000
  # Number of steps to take during evaluation.
  config.num_eval_steps = 20
  # Number of steps to generate predictions (used for BLEU score).
  # -1 will use the whole eval dataset.
  config.num_predict_steps = -1

  # Base learning rate.
  config.learning_rate = 0.0005

  # Linear learning rate warmup.
  config.warmup_steps = 4000

  # Cross entropy loss label smoothing.
  config.label_smoothing = 0.1

  # Decay factor for AdamW style weight decay.
  config.weight_decay = 0.0

  # Maximum length cutoff for training examples.
  config.max_target_length = 512 
  # Maximum length cutoff for eval examples.
  config.max_eval_target_length = 512 
  # Maximum length cutoff for predicted tokens.
  config.max_predict_length = 512 

  # whether use latent decoder
  config.latent = False 

  config.num_latent_tokens = 1

  # Inputs and targets share embedding.
  config.share_embeddings = True

  # Final logit transform uses embedding matrix transpose.
  config.logits_via_embedding = True

  # Number of transformer layers.
  config.num_layers = 6 

  # Size of query/key/value for attention.
  config.qkv_dim = 512 
  # Size of embeddings.
  config.emb_dim = 512 
  # Size of the MLP.
  config.mlp_dim = 2048  

  # Number of attention heads.
  config.num_heads = 8 

  # Dropout rate.
  config.dropout_rate = 0.1

  # Attention dropout rate.
  config.attention_dropout_rate = 0.1

  # Whether to save model checkpoints.
  config.save_checkpoints = True
  # Whether to restore from existing model checkpoints.
  config.restore_checkpoints = True
  # Save a checkpoint every these number of steps.
  config.checkpoint_freq = 3000

  # Use bfloat16 mixed precision training instead of float32.
  config.use_bfloat16 = True

  # Integer for PRNG random seed.
  config.seed = 0

  # Debug mode
  config.debug = False 

  return config
