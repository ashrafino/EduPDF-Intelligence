"""
Simple test to verify the project structure is correctly set up.
This can be run after installing dependencies.
"""

def test_imports():
    """Test that all modules can be imported."""
    try:
        # Test configuration system
        from config.settings import ConfigManager, AppConfig, SourceConfig
        print("✓ Configuration system imports successful")
        
        # Test module structure
        import scrapers
        import processors
        import filters
        import utils
        print("✓ All module directories accessible")
        
        # Test utilities
        from utils.logging_setup import setup_logging
        print("✓ Utility functions accessible")
        
        print("\n✅ Project structure verification complete!")
        print("All modules are properly organized and importable.")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure to install dependencies: pip install -r requirements.txt")


if __name__ == "__main__":
    test_imports()