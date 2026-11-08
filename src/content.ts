export type StepId = "embedding" | "qkv" | "attention" | "mlp" | "softmax" | "effect";

export type GuidePage = {
  id: string;
  title: string;
  body: string;
  focus: string | null;
  expand: StepId | null;
};

export const guide: GuidePage[] = [
  {
    id: "what",
    title: "What is a Vision Transformer?",
    body: "A Vision Transformer cuts an image into patches and runs the same block as a language Transformer. Each patch is a token. This page runs a real tiny model: 32×32 images, 8×8 patches, 4 blocks, 4 heads, width 32. ViT-Base uses the same diagram at 224×224, patch 16, 12 blocks, 12 heads, width 768.",
    focus: null,
    expand: null,
  },
  {
    id: "how",
    title: "How classification works",
    body: "Pick an example, paint the grid, or upload a picture. The model asks a different question from GPT: not “which word is next?” but “which class is this image?” The sampled class on the right is drawn from the probability bars. Sample again to draw a new one.",
    focus: "controls",
    expand: null,
  },
  {
    id: "architecture",
    title: "Three parts",
    body: "Embeddings turn patches into vectors. A stack of identical blocks mixes those vectors. A linear layer on the class token turns the last vector into one score per class, and softmax turns scores into probabilities.",
    focus: null,
    expand: null,
  },
  {
    id: "patches",
    title: "Patch embedding",
    body: "The 32×32 image is cut into a 4×4 grid of 8×8 patches. A convolution with stride 8 is a linear map on each patch: 3×8×8 = 192 numbers become a vector of 32. Patches never overlap.",
    focus: "embedding",
    expand: "embedding",
  },
  {
    id: "cls",
    title: "Class token",
    body: "A learned vector, CLS, is prepended. It is not a patch. Every block lets it read the patches, and the classifier reads only CLS at the end. Click CLS and watch which patches it uses.",
    focus: "embedding",
    expand: "embedding",
  },
  {
    id: "position",
    title: "Positional embedding",
    body: "Attention by itself has no notion of left or right. A learned vector is added to each token, one for CLS and one for each grid cell. After that add, the vector still has length 32.",
    focus: "embedding",
    expand: "embedding",
  },
  {
    id: "blocks",
    title: "The Transformer block",
    body: "Each block is layer norm, multi-head self-attention, a residual add, layer norm, an MLP, and another residual add. The next blocks repeat this. Step through them: the weights differ, the diagram does not.",
    focus: "block",
    expand: null,
  },
  {
    id: "qkv",
    title: "Query, key, value",
    body: "One linear layer maps each 32-vector to 96 numbers and splits them into query, key, and value. Those are split again across 4 heads, so each head holds 8 numbers per token. Query asks, key answers, value is what gets copied.",
    focus: "qkv",
    expand: "qkv",
  },
  {
    id: "attention",
    title: "Self-attention",
    body: "Each token scores every token, including itself. The score is the dot product of query and key, divided by √8. Softmax makes each row sum to 1. Those weights mix the values. Hover a cell: the image shows that pair.",
    focus: "attention",
    expand: "attention",
  },
  {
    id: "heads",
    title: "Multi-head",
    body: "Four heads run this matrix separately, then their outputs are concatenated back to 32 numbers and mixed by one projection. Switch heads. A head can lock onto color, edge, or the class token while another does something else.",
    focus: "attention",
    expand: "attention",
  },
  {
    id: "mask",
    title: "No causal mask",
    body: "GPT hides future words, so the upper triangle of its matrix is empty. A picture has no future. Every patch may read every other patch. Turn on the causal mask to force GPT’s rule and watch the class distribution fall apart.",
    focus: "attention",
    expand: "attention",
  },
  {
    id: "concat",
    title: "Output concatenation",
    body: "Head outputs of length 8 are placed side by side into a vector of 32. The projection matrix (32×32) mixes heads. After that, a token is one vector again.",
    focus: "attention",
    expand: "attention",
  },
  {
    id: "mlp",
    title: "MLP",
    body: "The MLP never mixes tokens. Each vector goes 32 → 128, through GELU, then 128 → 32. Attention decides who to read. The MLP decides what to store after reading.",
    focus: "mlp",
    expand: "mlp",
  },
  {
    id: "residual",
    title: "Residual",
    body: "The block adds the attention output onto the vector that entered the block, then adds the MLP output onto that. The original patch embedding still has a path to the end. Open the residual card to compare the size of the add and the skip.",
    focus: "block",
    expand: null,
  },
  {
    id: "norm",
    title: "Layer norm",
    body: "Before attention and before the MLP, each token is recentered and rescaled on its own 32 numbers, then a learned gain and bias are applied. Tokens are not mixed here.",
    focus: "qkv",
    expand: "qkv",
  },
  {
    id: "dropout",
    title: "Dropout",
    body: "During training, dropout zeros random coordinates and scales the rest by 1/(1−p). This run starts at p = 0, which is inference: every value is kept. Raise the rate and the probabilities jitter, because a different subset of coordinates is dropped.",
    focus: "mlp",
    expand: "mlp",
  },
  {
    id: "more",
    title: "Identical blocks",
    body: "The picture draws one block. The model runs four, each with its own weights. “More identical blocks” steps the activations you are looking at. Play walks the class token’s attention from the first block to the last.",
    focus: "more",
    expand: null,
  },
  {
    id: "logits",
    title: "Logits",
    body: "Only the class token is kept. A final layer norm, then a matrix (9×32), produces one score per class: three colors times three shapes. These scores are logits. They are not probabilities yet.",
    focus: "softmax",
    expand: "softmax",
  },
  {
    id: "probabilities",
    title: "Probabilities",
    body: "Softmax maps logits to positive numbers that sum to 1. The bars are those numbers. The outlined bar is the class just sampled. The darkest label under the controls is the most probable class, which can differ when sampling is broad.",
    focus: "softmax",
    expand: "softmax",
  },
  {
    id: "temperature",
    title: "Temperature",
    body: "Temperature divides the logits before softmax. Below 1 the largest class takes almost all the mass. Above 1 the bars flatten. The blocks do not rerun. Only this last distribution changes, the same place GPT’s temperature acts.",
    focus: "controls",
    expand: "softmax",
  },
  {
    id: "sampling",
    title: "Sampling",
    body: "Top-k keeps the k largest probabilities, renormalizes, and draws. Top-p keeps the smallest set whose mass reaches p, then draws. With nine classes this is the same operation as next-token sampling, on a tiny vocabulary.",
    focus: "controls",
    expand: "softmax",
  },
];

export const article: { heading: string; paragraphs: string[] }[] = [
  {
    heading: "The same block, on patches",
    paragraphs: [
      "The Transformer Explainer shows GPT-2 predicting the next word. This page shows a Vision Transformer predicting the class of an image. Between the input and the softmax, the operations match: a token vector, query-key-value, a square attention matrix, a residual, and a per-token MLP.",
      "The model on this page is small on purpose and it is real. It was trained on synthetic colored shapes and its validation accuracy is stored with the weights. Every bar, cell, and probability is a number from that forward pass.",
    ],
  },
  {
    heading: "Where the numbers live",
    paragraphs: [
      "An input is 3×32×32. Sixteen patches of 8×8×3 are each mapped to 32 dimensions, then a class token makes the sequence length 17. Query, key, and value expand that to 96 and split it into 4 heads of 8. Each head builds a 17×17 matrix. The MLP expands 32 to 128 and returns to 32. The classifier maps the class token’s 32 numbers to 9 logits.",
      "Attention is the only place tokens exchange information. The MLP, the layer norms, and the residual adds all act on one token at a time. That split is the thing to watch while you step through the blocks.",
    ],
  },
  {
    heading: "What a full ViT changes",
    paragraphs: [
      "ViT-Base, from Dosovitskiy et al. 2020, uses patch size 16 on a 224×224 image, so the sequence is 196 patches plus a class token. It stacks this block 12 times at width 768 with 12 heads. The matrices get larger. The operations stay the ones drawn here.",
      "Later vision models change pieces around this core. Swin limits attention to a window. MAE uses a bidirectional mask and asks the model to fill in missing patches. A detection model such as DETR keeps the block and adds cross-attention, where the query comes from a different sequence than the key. The causal mask on this page is the GPT rule, included so you can see what vision deliberately does not do.",
    ],
  },
];

export type PopoverId =
  | "patch"
  | "position"
  | "ln1"
  | "qkv"
  | "attention"
  | "concat"
  | "proj"
  | "residual1"
  | "ln2"
  | "gelu"
  | "mlp-up"
  | "mlp-down"
  | "residual2"
  | "dropout"
  | "classifier"
  | "softmax";

export const popovers: Record<PopoverId, { title: string; formula: string; body: string }> = {
  patch: {
    title: "Patch projection",
    formula: "Xₚ = Conv(image)    weight (32, 3, 8, 8)",
    body: "Each 8×8 patch is flattened to 192 numbers and multiplied by the same matrix. The bias has length 32. This is the token embedding of the Vision Transformer.",
  },
  position: {
    title: "Positional embedding",
    formula: "Z₀ = [CLS ; Xₚ] + Eₚₒₛ    Eₚₒₛ is (17, 32)",
    body: "Position is added, not concatenated. CLS uses row 0. Patch (r, c) uses row 1 + 4r + c. These vectors are learned, not a fixed sine.",
  },
  ln1: {
    title: "Layer norm",
    formula: "LN(x) = ((x − μ) / √(σ² + ε)) ⊙ γ + β",
    body: "μ and σ² are the mean and variance of that token’s 32 coordinates. ε is 1e-5. γ and β are learned. Other tokens do not enter this calculation.",
  },
  qkv: {
    title: "QKV weights",
    formula: "[Q K V] = LN(Z) Wqkv + b    Wqkv is (96, 32)",
    body: "One matrix produces query, key, and value stacked along the last dimension. The first 32 columns of the output are Q, the next 32 are K, the last 32 are V. Each of those is then split into 4 heads of 8.",
  },
  attention: {
    title: "Attention weights",
    formula: "A = softmax(Q Kᵀ / √8)    A is (17, 17) per head",
    body: "A row is a distribution over tokens. The row for the selected token is what gets painted on the image. Dividing by √8 keeps the dots from growing with the head width.",
  },
  concat: {
    title: "Concatenate heads",
    formula: "H = [head₁ ; head₂ ; head₃ ; head₄]    each head is (17, 8)",
    body: "Concatenation does not mix heads. It only places them side by side so the next projection sees all four.",
  },
  proj: {
    title: "Attention projection",
    formula: "O = H Wₒ + b    Wₒ is (32, 32)",
    body: "This is where heads mix. After it, each token is a single 32-vector again, ready for the residual add.",
  },
  residual1: {
    title: "Attention residual",
    formula: "Z′ = Z + Dropout(O)",
    body: "Z is the block input, before layer norm. The add gives every earlier representation a direct path forward. At dropout 0, Dropout(O) is O.",
  },
  ln2: {
    title: "Second layer norm",
    formula: "LN(Z′) before the MLP",
    body: "Same operation as the first layer norm, with its own γ and β. It runs on the residual stream after attention.",
  },
  gelu: {
    title: "GELU",
    formula: "GELU(x) = 0.5 x (1 + erf(x / √2))",
    body: "GELU is applied independently to every coordinate of the 128-wide hidden vector. It is smooth at zero, unlike ReLU, and it is the activation used in the GPT-2 explainer’s MLP as well.",
  },
  "mlp-up": {
    title: "MLP up",
    formula: "U = LN(Z′) W₁ + b    W₁ is (128, 32)",
    body: "The expansion gives the token room to store a feature that attention just assembled. No other token is visible to this matrix.",
  },
  "mlp-down": {
    title: "MLP down",
    formula: "D = GELU(U) W₂ + b    W₂ is (32, 128)",
    body: "The projection returns the token to width 32 so it can be added back onto the residual stream.",
  },
  residual2: {
    title: "MLP residual",
    formula: "Zₙₑₓₜ = Z′ + Dropout(D)",
    body: "This is the vector the next block receives. The last block’s Zₙₑₓₜ at position 0, the class token, is what the classifier sees after a final layer norm.",
  },
  dropout: {
    title: "Dropout",
    formula: "keep x / (1 − p) with probability 1 − p, else 0",
    body: "The mask is drawn per coordinate of the attention projection and of the MLP output. p = 0 leaves the trained model untouched. A new draw uses a new seed, so the same image can produce a different distribution.",
  },
  classifier: {
    title: "Classification head",
    formula: "logits = LN(CLS) Wₕ + b    Wₕ is (9, 32)",
    body: "Nine classes are three colors crossed with three shapes. Only the class token is multiplied. Patch tokens are discarded after the last block.",
  },
  softmax: {
    title: "Softmax",
    formula: "pᵢ = exp(zᵢ / T) / Σⱼ exp(zⱼ / T)",
    body: "T is the temperature. Sampling then keeps a top-k or top-p subset of p, renormalizes that subset, and draws an index. The bars always show the full softmax, before that cut.",
  },
};
