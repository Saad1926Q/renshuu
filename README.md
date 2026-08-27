<h1 align="center">mini-moe</h1>

<p align="center">
Trying to build a sparse Mixture-of-Experts layer from scratch to understand how MoEs work
</p>

---

## Motivation

I was learning about Mixture-of-Experts models by reading this [MoE guide](https://sodakeyeatsmush.vercel.app/moe), which I had prepared using Codex.

After reading about the ideas behind MoEs, I wanted to try some of them out practically by coding a small MoE layer in PyTorch.

## Goal

The rough goal of this project is to understand what happens inside an MoE layer: how tokens are routed, how only the selected experts process each token, and how the different expert outputs are combined.

This is not meant to be a complete language model or a production-ready MoE implementation. It is a small implementation focused on learning the core mechanics.

## What Has Been Implemented

- A bias-free SwiGLU expert
- A learned router that assigns tokens to the top-k experts
- Shared experts that process every token
- Routed experts that process only their assigned tokens
- Weighted aggregation of routed-expert outputs
- Routing outputs including logits, selected expert indices, routing weights, and expert assignment counts
