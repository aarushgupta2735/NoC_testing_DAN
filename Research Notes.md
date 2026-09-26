# A DAN inspired attempt at solving NoC testing

# Theory:
1. Why preemption? What not have it 1 or 0? What's the tradeoff?
    Large p : less subtasks (less total overhead) but more conflicts with other tests
    Small p: more subtasks (more total overhead) but less conflicts with other tests

Core specific properties that we must add for better preemption value calculation: 

Patterns (p_count): Workload in the test sequence given to the core
Scan: Fixed latency/subtask

# Current Main State
We have the lmga_igsa code as the starter code to build on top of : https://github.com/aarushgupta2735/NoC_testing_LMGA_IGSA 

# Objective: 
1. Create a Multi-Head Attention based solution for mapping IO pairs to cores 
2. Find preemption values for a particular IO channel (must take in conflicts from other core tests)
The pattern scheduling is done by SFT scheduling. 

# Architecture: 
We split the problem across objectives in two phases (mapping and preemptive values).

## Inputs:
    1. Take the embeggings for the mesh topology from a GNN
    2. Core embeddings will have (x,y,pattern,scan) as input for each core

## Model v1.0:
    1. Multi Head Attention that takes concatenation of the all the core embeddings(transformed with a WQ) as Q, and the IO pairs embedding (Transformed with a WK) as the K and V. Softmax over scores to find the core for a given IO. 
    
    2. To find the preemption value, I not only want to give the model the IO embedding but also information of the cores that have been mapped to it. For this we use another attention layer that can allow the model to look back at its alloted cores and give scores. These scores would be used for performing a weighted average over the core embeddings for each core (transformer style) that was mapped to the particular IO pair (this can be optimised by using step 1 values again or interveaving both). The final vector is then concatenated with the IO pair and then fed to a MLP for preemptive value. 

## Model v1.1:
    1. Stays as it is.

    2. Preumption value not only depends upon the allocated cores to the particular IO pair but more on the conflicts with other tests. That information must be given for this decision. After phase 1, we will be computing this conflict embedding using a GNN by reusing the check_path_conflict in simulator.py on the computed IO-Core mapping. We cache these values to use later during SFT to reduce added time complexity. We will be combining this conflict embedding with the per-core embedding with an MLP directly instead of weighted average one in v1.0 to get the preemptive value. This assumes that we have already given all the conflict related information with the conflict embedding.  

## Model v2:

Per-core input features: [row, col, patterns, scan] (4-dim)

### Phase 1 (mapping): 
all cores' features → linear projection → Q; all IOs' embeddings → linear projections → K, V; multi-head cross-attention, softmax over IOs per core; hard sample one IO per core. This retires the LSTM encoder/decoder entirely — no recurrence, single parallel attention pass.

### Intermediate: Conflict table (shared utility, data.py): 
Given a resolved mapping, compute the full N*N pairwise check_path_conflict table once. Passed to both phase 2 and the simulator.

### Phase 2 (preemption), per-core:

Conflict-GNN: a GAT over the conflict-table-derived graph, operating on raw core features (position+patterns+scan) independently from phase 1 — produces a g_c context vector per core.
MLP(concat(core_embedding, g_c)) → preemption bucket logits, per core. No attention/weighted-average step.

### Simulator:
Accepts an optional precomputed conflict table; looks up instead of recomputing check_path_conflict inline.

### Training: 
- No detach anywhere (per our earlier, confirmed decision) — both phases' log-probs backprop through the shared core-feature projections.
- Reinforce

### PreTraining (V2_pretrain branch) :
In v1, we were using pretraining to train our IO-mapping phase without phase 2 and GA. 
v2 was not using pretraining but this can have quite a lot of problems as REINFORCE can be unstable. 
Therefore we are creating a configurable option to pretrain our IO-mapper with a known preumption head assumed (hyperparameter). Unlike v1's pretraining, we plan to also use GA's new samples during pretraining as this would allow the model to explore more unlike v1 (the only exploration is sampling from its ploicy and not argmax).

## Open Questions:
- Why one preemption value for a IO pair ? Why a particular constant for division?

# Potential Ideas for Later:

Q. What if you weighted the mesh topology with p_count after mapping from phase 1?

1. reinforce.py 
    -> try changing EMA to non discounted baseline

    Result:

    -> switch reinforce with PPO 

    Result:

## Other potential directions

### Multi-Agent Path Finding : 
1. "Multi-Agent Deep Reinforcement Learning (MADRL) has emerged as a primary focus for decentralized conflict resolution (Orr & Dutta, 2023)." 
2. To address conflicting paths in constrained environments, researchers frequently combine RL with Imitation Learning (IL). The PRIMAL framework, introduced by Sartoretti et al. (2019), trains agents to reactively plan paths while imitating centralized expert behavior. This hybrid approach enables implicit coordination in partially-observable environments without requiring explicit agent-to-agent communication.
3. Ma et al. (2021) developed a deep Q-network framework named Distributed Heuristic Communication (DHC)

Reference for this:
Bello, I., Pham, H., Le, Q. V., Norouzi, M., & Bengio, S. (2016). Neural Combinatorial Optimization with Reinforcement Learning. arXiv. https://doi.org/10.48550/arxiv.1611.09940
Cited by: 2961

Kool, W., van Hoof, H., & Welling, M. (2018). Attention, Learn to Solve Routing Problems! arXiv. https://doi.org/10.48550/arxiv.1803.08475
Cited by: 2935

Ma, Z., Luo, Y., & Ma, H. (2021). Distributed Heuristic Multi-Agent Path Finding with Communication. 2021 IEEE International Conference on Robotics and Automation (ICRA), 8699–8705. https://doi.org/10.1109/icra48506.2021.9560748
Cited by: 208

Mazyavkina, N., Sviridov, S., Ivanov, S., & Burnaev, E. (2021). Reinforcement learning for combinatorial optimization: A survey. Computers & Operations Research, 134, 105400. https://doi.org/10.1016/j.cor.2021.105400
Cited by: 1188

Orr, J., & Dutta, A. (2023). Multi-Agent Deep Reinforcement Learning for Multi-Robot Applications: A Survey. Sensors, 23(7), 3625. https://doi.org/10.3390/s23073625
Cited by: 337

Sartoretti, G., Kerr, J., Shi, Y., Wagner, G., Kumar, T. K. S., Koenig, S., & Choset, H. (2019). PRIMAL: Pathfinding via Reinforcement and Imitation Multi-Agent Learning. IEEE Robotics and Automation Letters, 4, 2378–2385. https://doi.org/10.1109/lra.2019.2903261
Cited by: 712

Zhang, C., Song, W., Cao, Z., Zhang, J., Tan, P. S., & Xu, C. (2020). Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning. arXiv. https://doi.org/10.48550/arxiv.2010.12367
Cited by: 736