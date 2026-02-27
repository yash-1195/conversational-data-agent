from src.core.config_loader import load_config

def test_config_loads():
    config = load_config()
    assert config.llm.temperature >= 0.0
    assert config.data.max_upload_size_mb == 50