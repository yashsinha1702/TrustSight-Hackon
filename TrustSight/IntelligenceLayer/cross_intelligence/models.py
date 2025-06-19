import networkx as nx
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Set, Optional, Any
from .enums import SignalType, NetworkType

@dataclass
class FraudSignal:
    """Data model for fraud detection signals that trigger network investigations"""
    signal_id: str
    signal_type: SignalType
    entity_id: str
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_detector: str = ""

@dataclass
class NetworkNode:
    """Data model for nodes in the fraud network graph"""
    node_id: str
    node_type: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    fraud_score: float = 0.0
    investigation_depth: int = 0
    connected_nodes: Set[str] = field(default_factory=set)

@dataclass
class NetworkEdge:
    """Data model for edges connecting nodes in the fraud network"""
    source: str
    target: str
    edge_type: str
    weight: float = 1.0
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Investigation:
    """Data model for fraud network investigations"""
    investigation_id: str
    initial_signal: FraudSignal
    start_time: datetime
    network_graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    nodes_investigated: Set[str] = field(default_factory=set)
    expansion_path: List[Dict] = field(default_factory=list)
    network_type: Optional[NetworkType] = None
    financial_impact: float = 0.0
    key_players: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    status: str = "in_progress"