def pytest_configure(config):
    config.addinivalue_line("markers", "optional: needs an optional package (for example the upstream CaMeL package); skipped when it is missing")
