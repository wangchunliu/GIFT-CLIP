from graph_image_interaction import encode_pair_with_interaction
from attention_visualizer import plot_attention_heatmap

clip_model.eval(); myTransformer.eval(); interaction.eval()

img = ...   # (1, 3, H, W)
text = ...  # (1, L) tokenized

T_feat, I_feat, (a_I, a_G), (alpha, beta) = encode_pair_with_interaction(
    clip_model, interaction, img, text,
    head_inputs, relation_inputs, tail_inputs,
    0, attention_mask,
    myTransformer,
    img_to_tokens, text_to_tokens, graph_to_tokens,
    alpha_module, beta_module,
    return_attn=True)

# a_I: (B, h, 1, M)  -- graph -> image  (这里只有 1 个 image token)
# a_G: (B, h, M, 1)  -- image -> graph

# Image -> Graph 注意力
plot_attention_heatmap(
    a_G,                                # (1, h, M, 1)
    query_labels=[f"node_{i}" for i in range(a_G.size(2))],
    key_labels=["img_token"],
    head_idx=0,
    title="Image -> Graph Attention",
    save_path="vis/img_to_graph.png")

# Graph -> Image 注意力
plot_attention_heatmap(
    a_I.squeeze(-1).unsqueeze(-1),     # 把它当作 (1, h, M) -> (1, h, M, 1)
    query_labels=["img_token"],
    key_labels=[f"node_{i}" for i in range(a_I.size(-1))],
    head_idx=0,
    title="Graph -> Image Attention",
    save_path="vis/graph_to_img.png")
