import os

from backend.shared.paths import PROJECT_ROOT


class TestProjectRoot:
    def test_points_to_directory_containing_docker_compose(self):
        assert os.path.exists(os.path.join(PROJECT_ROOT, "docker-compose.yml"))

    def test_is_absolute_path(self):
        assert os.path.isabs(PROJECT_ROOT)
