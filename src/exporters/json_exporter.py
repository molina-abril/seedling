"""JSON exporter for Cluster, Paper, RunState, and other models."""

import json
from pathlib import Path
from typing import List, Dict, Any

from src.models.cluster import Cluster
from src.models.paper import Paper
from src.models.run_state import RunState


class ExportManager:
    """Unified export manager for JSON serialization of model objects."""
    
    @staticmethod
    def export_clusters(
        clusters: List[Cluster],
        output_path: str,
        verbose: bool = True
    ) -> str:
        """
        Export clusters to JSON file.
        
        Args:
            clusters: List of Cluster objects
            output_path: Path to output JSON file
            verbose: Print progress
            
        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        clusters_data = [
            cluster.model_dump(mode='python')
            for cluster in clusters
        ]

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(clusters_data, f, indent=2, ensure_ascii=False)
        
        if verbose:
            print(f"[Export] ✓ Exported {len(clusters)} clusters to {output_path}")
        
        return str(output_path)
    
    @staticmethod
    def export_papers(
        papers: List[Paper],
        output_path: str,
        verbose: bool = True
    ) -> str:
        """
        Export papers to JSON file.
        
        Args:
            papers: List of Paper objects
            output_path: Path to output JSON file
            verbose: Print progress
            
        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        papers_data = [
            paper.model_dump(mode='python')
            for paper in papers
        ]

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(papers_data, f, indent=2, ensure_ascii=False, default=str)
        
        if verbose:
            print(f"[Export] ✓ Exported {len(papers)} papers to {output_path}")
        
        return str(output_path)
    
    @staticmethod
    def export_metrics(
        metrics: Dict[str, Any],
        output_path: str,
        verbose: bool = True
    ) -> str:
        """
        Export metrics to JSON file.
        
        Args:
            metrics: Dictionary of metrics
            output_path: Path to output JSON file
            verbose: Print progress
            
        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
        
        if verbose:
            print(f"[Export] ✓ Exported metrics to {output_path}")
        
        return str(output_path)
    
    @staticmethod
    def export_run_state(
        run_state: RunState,
        output_path: str,
        verbose: bool = True
    ) -> str:
        """
        Export complete RunState to JSON file.
        
        Args:
            run_state: RunState object
            output_path: Path to output JSON file
            verbose: Print progress
            
        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        run_state_data = run_state.model_dump(mode='python')

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(run_state_data, f, indent=2, ensure_ascii=False, default=str)
        
        if verbose:
            print(f"[Export] ✓ Exported RunState to {output_path}")
        
        return str(output_path)
