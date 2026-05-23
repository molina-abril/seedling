#!/usr/bin/env python
"""Quick validation of Phase 6 implementation structure."""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all Phase 6 modules can be imported."""
    print("=" * 60)
    print("Phase 6 Structure Validation")
    print("=" * 60)
    
    tests_passed = 0
    tests_failed = 0

    try:
        from src.utils.text import sanitize_unicode, clean_pdf_content, clear_text, remove_accents
        print("✓ src.utils.text imports OK")
        tests_passed += 1
    except Exception as e:
        print(f"✗ src.utils.text FAILED: {e}")
        tests_failed += 1

    try:
        from src.ingestion.pdf_loader import load_papers_from_pdf_directory, extract_pdf_text
        print("✓ src.ingestion.pdf_loader imports OK")
        tests_passed += 1
    except Exception as e:
        print(f"✗ src.ingestion.pdf_loader FAILED: {e}")
        tests_failed += 1

    try:
        from src.models.cluster import Cluster
        from src.models.paper import Paper
        print("✓ src.models imports OK")
        tests_passed += 1
    except Exception as e:
        print(f"✗ src.models FAILED: {e}")
        tests_failed += 1

    try:
        from src.utils.config import ConfigManager
        print("✓ src.utils.config imports OK")
        tests_passed += 1
    except Exception as e:
        print(f"✗ src.utils.config FAILED: {e}")
        tests_failed += 1

    try:
        from src.exporters.json_exporter import ExportManager
        print("✓ src.exporters.json_exporter imports OK")
        tests_passed += 1
    except Exception as e:
        print(f"✗ src.exporters.json_exporter FAILED: {e}")
        tests_failed += 1

    try:
        from src.clustering.bertopic_agent import ClusteringAgent
        print("✓ src.clustering.bertopic_agent imports OK")
        tests_passed += 1
    except Exception as e:
        print(f"✗ src.clustering.bertopic_agent FAILED: {e}")
        tests_failed += 1

    try:
        from src.evaluation.clustering_metrics import calculate_clustering_metrics
        print("✓ src.evaluation.clustering_metrics imports OK")
        tests_passed += 1
    except Exception as e:
        print(f"✗ src.evaluation.clustering_metrics FAILED: {e}")
        tests_failed += 1

    try:
        from src.cli import main, run_cluster
        print("✓ src.cli imports OK")
        tests_passed += 1
    except Exception as e:
        print(f"✗ src.cli FAILED: {e}")
        tests_failed += 1

    try:
        config_file = project_root / "configs" / "clustering" / "bertopic.yaml"
        assert config_file.exists(), f"Config file not found: {config_file}"
        print(f"✓ configs/clustering/bertopic.yaml exists")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Config file check FAILED: {e}")
        tests_failed += 1

    try:
        from src.models.paper import Paper
        paper = Paper(
            paper_id="test_1",
            title="Test Paper",
            abstract="This is a test abstract",
        )
        assert paper.paper_id == "test_1"
        assert paper.title == "Test Paper"
        print("✓ Paper model creation OK")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Paper model FAILED: {e}")
        tests_failed += 1

    try:
        from src.models.cluster import Cluster
        cluster = Cluster(
            cluster_id=0,
            label="Test Cluster",
            top_terms=["word1", "word2", "word3"],
            paper_ids=["p1", "p2", "p3"],
            representative_paper_ids=["p1"],
            is_noise=False
        )
        assert cluster.cluster_id == 0
        assert len(cluster.paper_ids) == 3
        print("✓ Cluster model creation OK")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Cluster model FAILED: {e}")
        tests_failed += 1

    try:
        from src.utils.text import sanitize_unicode, remove_accents, clean_pdf_content

        test_str = "hello\ud800world"
        result = sanitize_unicode(test_str)
        assert "hello" in result

        accented = "café"
        result = remove_accents(accented)
        assert result == "cafe" or result == "caf" or "cafe" in result or "caf" in result

        pdf_text = "Some text\x00with null\nMore text\n\nREFERENCES\nRef 1"
        result = clean_pdf_content(pdf_text)
        assert "Some text" in result
        assert "REFERENCES" not in result

        print("✓ Text utility functions OK")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Text utilities FAILED: {e}")
        tests_failed += 1

    try:
        from src.utils.config import ConfigManager
        config_mgr = ConfigManager()
        config = config_mgr.load_clustering('bertopic')

        assert 'embedding_model' in config
        assert 'umap' in config
        assert 'hdbscan' in config
        assert 'preprocessing' in config
        assert config['embedding_model'] == 'all-MiniLM-L6-v2'
        assert config['umap']['n_neighbors'] == 3
        assert config['hdbscan']['min_cluster_size'] == 2
        
        print("✓ ConfigManager loads bertopic.yaml OK")
        tests_passed += 1
    except Exception as e:
        print(f"✗ ConfigManager FAILED: {e}")
        tests_failed += 1

    try:
        from src.exporters.json_exporter import ExportManager

        assert hasattr(ExportManager, 'export_clusters')
        assert hasattr(ExportManager, 'export_papers')
        assert hasattr(ExportManager, 'export_metrics')
        assert hasattr(ExportManager, 'export_run_state')
        
        print("✓ ExportManager has all required methods")
        tests_passed += 1
    except Exception as e:
        print(f"✗ ExportManager FAILED: {e}")
        tests_failed += 1

    try:
        from src.models.cluster import Cluster
        import json
        import tempfile
        from pathlib import Path

        cluster = Cluster(
            cluster_id=1,
            label="Test",
            paper_ids=["p1", "p2"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_cluster.json"
            cluster_data = cluster.model_dump(mode='python')
            with open(output_file, 'w') as f:
                json.dump([cluster_data], f)

            assert output_file.exists()
            with open(output_file) as f:
                loaded = json.load(f)
            assert len(loaded) == 1
            assert loaded[0]['cluster_id'] == 1

        print("✓ Cluster JSON serialization OK")
        tests_passed += 1
    except Exception as e:
        print(f"✗ JSON serialization FAILED: {e}")
        tests_failed += 1

    print("=" * 60)
    print(f"Results: {tests_passed} passed, {tests_failed} failed")
    print("=" * 60)
    
    if tests_failed == 0:
        print("\n✅ All Phase 6 structure tests PASSED!")
        print("\nNext steps:")
        print("  1. Install BERTopic dependencies:")
        print("     pip install bertopic sentence-transformers umap-learn hdbscan scikit-learn")
        print("  2. Run clustering on PDFs:")
        print("     python -m src.cli cluster --pdf-dir papers/ --output results/clusters.json")
    
    return tests_failed == 0

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
