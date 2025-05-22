import os
import tempfile
from collections import OrderedDict
from unittest import mock

import pytest
import yaml

from clarifai.utils.config import Config, Context


@pytest.fixture
def temp_config_file():
    """Create a temporary config file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        test_config = {
            'current_context': 'test_context',
            'contexts': {
                'test_context': {
                    'CLARIFAI_USER_ID': 'test_user',
                    'CLARIFAI_PAT': 'test_pat',
                    'CLARIFAI_API_BASE': 'https://api.test.com'
                },
                'alternate_context': {
                    'CLARIFAI_USER_ID': 'alternate_user',
                    'CLARIFAI_PAT': 'alternate_pat',
                    'CLARIFAI_API_BASE': 'https://api.alternate.com',
                    'CLARIFAI_COMPUTE_CLUSTER_ID': 'cluster123'
                }
            }
        }
        yaml.dump(test_config, f)
        file_name = f.name
    
    yield file_name
    
    try:
        os.unlink(file_name)
    except OSError:
        pass


class TestContext:
    """Tests for the Context class."""
    
    def test_context_init_with_env(self):
        """Test Context initialization with 'env' parameter."""
        env = {'CLARIFAI_USER_ID': 'test_user', 'CLARIFAI_PAT': 'test_pat'}
        context = Context('test_context', env=env)
        
        assert context['name'] == 'test_context'
        assert context['env'] == env
    
    def test_context_init_with_kwargs(self):
        """Test Context initialization with key-value parameters."""
        context = Context('test_context', CLARIFAI_USER_ID='test_user', CLARIFAI_PAT='test_pat')
        
        assert context['name'] == 'test_context'
        assert context['env'] == {'CLARIFAI_USER_ID': 'test_user', 'CLARIFAI_PAT': 'test_pat'}
    
    def test_context_getattr(self):
        """Test accessing attributes via __getattr__ method."""
        context = Context('test_context', CLARIFAI_USER_ID='test_user', CLARIFAI_PAT='test_pat')
        
        assert context.user_id == 'test_user'
        assert context.pat == 'test_pat'
        
        with pytest.raises(AttributeError):
            context.nonexistent_attr
    
    def test_context_getattr_with_envvar(self):
        """Test accessing attributes with ENVVAR designation."""
        context = Context('test_context', CLARIFAI_USER_ID='ENVVAR', CLARIFAI_PAT='test_pat')
        
        with mock.patch.dict(os.environ, {'CLARIFAI_USER_ID': 'env_user'}):
            assert context.user_id == 'env_user'
        
        # Remove CLARIFAI_USER_ID from environment and try to access it
        with mock.patch.dict(os.environ, {}, clear=True):
            with pytest.raises(AttributeError):
                print(context.user_id)
    
    def test_context_setattr(self):
        """Test setting attributes."""
        context = Context('test_context')
        
        context.user_id = 'new_user'
        assert context['env']['user_id'] == 'new_user'
        
        context.CLARIFAI_PAT = 'new_pat'
        assert context['env']['CLARIFAI_PAT'] == 'new_pat'
    
    def test_context_hasattr(self):
        """Test hasattr functionality."""
        context = Context('test_context', CLARIFAI_USER_ID='test_user')
        
        assert hasattr(context, 'name')
        assert hasattr(context, 'user_id')
        assert not hasattr(context, 'nonexistent_attr')
    
    def test_context_delattr(self):
        """Test deleting attributes."""
        context = Context('test_context', CLARIFAI_USER_ID='test_user')
        
        # The attribute should be deleted using the key name in env
        delattr(context, 'CLARIFAI_USER_ID')
        assert 'CLARIFAI_USER_ID' not in context['env']
        
        with pytest.raises(AttributeError):
            delattr(context, 'nonexistent_attr')
    
    def test_to_serializable_dict(self):
        """Test conversion to serializable dict."""
        context = Context('test_context', CLARIFAI_USER_ID='test_user', CLARIFAI_PAT='test_pat')
        
        serialized = context.to_serializable_dict()
        assert serialized == {'CLARIFAI_USER_ID': 'test_user', 'CLARIFAI_PAT': 'test_pat'}
    
    def test_set_to_env(self):
        """Test setting context values to environment variables."""
        context = Context('test_context', CLARIFAI_USER_ID='test_user', key='value')
        
        # Clear existing env vars to ensure clean test
        if 'CLARIFAI_USER_ID' in os.environ:
            del os.environ['CLARIFAI_USER_ID']
        if 'CLARIFAI_KEY' in os.environ:
            del os.environ['CLARIFAI_KEY']
        
        context.set_to_env()
        
        assert os.environ['CLARIFAI_USER_ID'] == 'test_user'
        assert os.environ['CLARIFAI_KEY'] == 'value'


class TestConfig:
    """Tests for the Config class."""
    
    def test_config_from_yaml(self, temp_config_file):
        """Test loading config from YAML file."""
        config = Config.from_yaml(temp_config_file)
        
        assert config.current_context == 'test_context'
        assert config.filename == temp_config_file
        assert len(config.contexts) == 2
        assert 'test_context' in config.contexts
        assert 'alternate_context' in config.contexts
    
    def test_config_to_dict(self, temp_config_file):
        """Test conversion to dictionary."""
        config = Config.from_yaml(temp_config_file)
        
        result = config.to_dict()
        assert result['current_context'] == 'test_context'
        assert 'contexts' in result
        assert 'test_context' in result['contexts']
        assert result['contexts']['test_context']['CLARIFAI_USER_ID'] == 'test_user'
    
    def test_config_to_yaml(self, temp_config_file):
        """Test saving to YAML file."""
        config = Config.from_yaml(temp_config_file)
        
        # Create a new temporary file for output
        with tempfile.NamedTemporaryFile(delete=False) as temp_out:
            out_file = temp_out.name
        
        config.to_yaml(out_file)
        
        # Read the file back and verify contents
        with open(out_file, 'r') as f:
            saved_config = yaml.safe_load(f)
        
        assert saved_config['current_context'] == 'test_context'
        assert 'test_context' in saved_config['contexts']
        assert saved_config['contexts']['test_context']['CLARIFAI_USER_ID'] == 'test_user'
        
        # Clean up
        os.unlink(out_file)
    
    def test_config_current_property(self, temp_config_file):
        """Test the current property that returns the current context."""
        config = Config.from_yaml(temp_config_file)
        
        current = config.current
        assert current['name'] == 'test_context'
        assert current.user_id == 'test_user'
        
        # Change current context and verify
        config.current_context = 'alternate_context'
        current = config.current
        assert current['name'] == 'alternate_context'
        assert current.user_id == 'alternate_user'
        assert current.compute_cluster_id == 'cluster123'
    
    def test_post_init(self):
        """Test the __post_init__ method."""
        contexts = OrderedDict({
            'test_context': {'CLARIFAI_USER_ID': 'test_user'},
            'no_name_context': {'name': 'with_name', 'CLARIFAI_USER_ID': 'other_user'}
        })
        
        config = Config(current_context='test_context', filename='dummy.yaml', contexts=contexts)
        
        assert config.contexts['test_context']['name'] == 'test_context'
        assert config.contexts['no_name_context']['name'] == 'with_name'
        assert isinstance(config.contexts['test_context'], Context)
        assert isinstance(config.contexts['no_name_context'], Context)