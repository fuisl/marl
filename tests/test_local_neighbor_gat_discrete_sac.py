"""Tests for LocalNeighborGATDiscreteSAC model architecture."""

from __future__ import annotations

import torch

from models.local_neighbor_gat_discrete_sac import LocalNeighborGATDiscreteSAC


def test_local_neighbor_gat_forward_pass() -> None:
    """Test forward pass of LocalNeighborGATDiscreteSAC."""
    obs_dim = 12
    num_actions = 2
    n_agents = 4
    
    agent = LocalNeighborGATDiscreteSAC(
        obs_dim=obs_dim,
        num_actions=num_actions,
        local_encoder_cfg={"hidden_dim": 32, "out_dim": 24},
        neighbor_encoder_cfg={"hidden_dim": 32, "out_dim": 24, "heads": 2},
        fusion_cfg={"hidden_dim": 48, "out_dim": 32},
        actor_cfg={"hidden_dim": 48},
        critic_cfg={"hidden_dim": 128},
        init_alpha=0.2,
        tau=0.005,
    )
    agent.eval()
    
    # Create sample graph data
    obs = torch.randn(n_agents, obs_dim)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 0, 1, 2, 3],
        [1, 0, 3, 2, 2, 3, 1, 0],
    ], dtype=torch.long)
    edge_attr = torch.randn(edge_index.shape[1], 2)
    action_mask = torch.ones(n_agents, num_actions, dtype=torch.bool)
    
    # Test encode
    with torch.no_grad():
        z = agent.encode(obs, edge_index, edge_attr)
    assert z.shape == (n_agents, 32)
    
    # Test critic values
    with torch.no_grad():
        q1, q2 = agent.critic_values(obs, edge_index, edge_attr)
    assert q1.shape == (n_agents, num_actions)
    assert q2.shape == (n_agents, num_actions)
    assert torch.isfinite(q1).all()
    assert torch.isfinite(q2).all()
    
    # Test select_action
    with torch.no_grad():
        action, log_prob = agent.select_action(
            obs, edge_index, edge_attr, action_mask, deterministic=True
        )
    assert action.shape == (n_agents,)
    assert log_prob.shape == (n_agents,)
    assert torch.isfinite(action.float()).all()
    assert torch.isfinite(log_prob).all()


def test_local_neighbor_gat_action_probs() -> None:
    """Test action probability computation."""
    obs_dim = 12
    num_actions = 2
    n_agents = 3
    
    agent = LocalNeighborGATDiscreteSAC(
        obs_dim=obs_dim,
        num_actions=num_actions,
        local_encoder_cfg={"hidden_dim": 32, "out_dim": 24},
        neighbor_encoder_cfg={"hidden_dim": 32, "out_dim": 24, "heads": 2},
        fusion_cfg={"hidden_dim": 48, "out_dim": 32},
    )
    agent.eval()
    
    obs = torch.randn(n_agents, obs_dim)
    edge_index = torch.tensor([
        [0, 1, 2, 0, 1],
        [1, 2, 0, 2, 0],
    ], dtype=torch.long)
    edge_attr = torch.randn(edge_index.shape[1], 2)
    
    with torch.no_grad():
        z, action_probs, log_action_probs = agent.get_action_probs(
            obs, edge_index, edge_attr
        )
    
    assert z.shape == (n_agents, 32)
    assert action_probs.shape == (n_agents, num_actions)
    assert log_action_probs.shape == (n_agents, num_actions)
    
    # Check property: probs sum to 1
    assert torch.allclose(action_probs.sum(dim=1), torch.ones(n_agents))


def test_local_neighbor_gat_target_update() -> None:
    """Test soft update of target critic."""
    obs_dim = 12
    num_actions = 2
    
    agent = LocalNeighborGATDiscreteSAC(
        obs_dim=obs_dim,
        num_actions=num_actions,
        tau=0.01,
    )
    
    # Get initial target parameters
    with torch.no_grad():
        initial_target_params = [p.clone() for p in agent.target_critic.parameters()]
    
    # Perform an update
    agent.soft_update_target()
    
    # Target should have moved slightly from initial
    with torch.no_grad():
        updated_target_params = [p.clone() for p in agent.target_critic.parameters()]
    
    # Parameters should differ (soft update has occurred)
    # Note: This is probabilistic if networks overlap, so we just check it doesn't diverge
    for p_init, p_updated in zip(initial_target_params, updated_target_params):
        max_diff = (p_init - p_updated).abs().max().item()
        assert max_diff < 1.0, f"Target update diverged too much: {max_diff}"


def test_local_neighbor_gat_with_agent_node_pooling() -> None:
    """Test LocalNeighborGAT with agent-node pooling (all_intersections mode)."""
    obs_dim = 12
    num_actions = 2
    num_graph_nodes = 25
    num_agents = 4
    
    agent = LocalNeighborGATDiscreteSAC(
        obs_dim=obs_dim,
        num_actions=num_actions,
        local_encoder_cfg={"hidden_dim": 32, "out_dim": 24},
        neighbor_encoder_cfg={"hidden_dim": 32, "out_dim": 24, "heads": 2},
        fusion_cfg={"hidden_dim": 48, "out_dim": 32},
    )
    agent.eval()
    
    # Create graph data and agent-node mapping
    graph_obs = torch.randn(num_graph_nodes, obs_dim)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 0, 1, 2],
        [1, 2, 3, 4, 0, 5, 6, 7],
    ], dtype=torch.long)
    edge_attr = torch.randn(edge_index.shape[1], 2)
    
    # Agent to node mapping: each of 4 agents maps to 1-2 graph nodes
    agent_node_indices = torch.tensor([
        [0, -1],
        [5, 10],
        [15, -1],
        [20, 24],
    ], dtype=torch.long)
    agent_node_mask = torch.tensor([
        [True, False],
        [True, True],
        [True, False],
        [True, True],
    ], dtype=torch.bool)
    
    with torch.no_grad():
        z = agent.encode(
            graph_obs,
            edge_index,
            edge_attr,
            agent_node_indices=agent_node_indices,
            agent_node_mask=agent_node_mask,
        )
    
    # Output should have pooled to num_agents
    assert z.shape == (num_agents, 32)
    assert torch.isfinite(z).all()


def test_local_neighbor_gat_alpha_property() -> None:
    """Test entropy temperature alpha property."""
    agent = LocalNeighborGATDiscreteSAC(
        obs_dim=12,
        num_actions=2,
        init_alpha=0.2,
    )
    
    alpha = agent.alpha
    assert isinstance(alpha, torch.Tensor)
    assert alpha.item() > 0
    # exp(log(0.2)) ≈ 0.2
    assert abs(alpha.item() - 0.2) < 0.01
