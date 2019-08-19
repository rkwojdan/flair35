
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import pytest
try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path


@pytest.fixture(scope=u'module')
def resources_path():
    return (Path(__file__).parent / u'resources')


@pytest.fixture(scope=u'module')
def tasks_base_path(resources_path):
    return (resources_path / u'tasks')


@pytest.fixture(scope=u'module')
def results_base_path(resources_path):
    return (resources_path / u'results')


def pytest_addoption(parser):
    parser.addoption(u'--runslow', action=u'store_true',
                     default=False, help=u'run slow tests')
    parser.addoption(u'--runintegration', action=u'store_true',
                     default=False, help=u'run integration tests')


def pytest_collection_modifyitems(config, items):
    if (config.getoption(u'--runslow') and config.getoption(u'--runintegration')):
        return
    if (not config.getoption(u'--runslow')):
        skip_slow = pytest.mark.skip(reason=u'need --runslow option to run')
        for item in items:
            if (u'slow' in item.keywords):
                item.add_marker(skip_slow)
    if (not config.getoption(u'--runintegration')):
        skip_integration = pytest.mark.skip(
            reason=u'need --runintegration option to run')
        for item in items:
            if (u'integration' in item.keywords):
                item.add_marker(skip_integration)
